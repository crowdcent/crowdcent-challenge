"""Load a model zoo model and predict with it.

```python
from crowdcent_challenge import ChallengeClient, zoo

model = zoo.load("hl-ridge-baseline", client=ChallengeClient())
predictions = model.predict(inference_data)
```

Each ``kind`` loads from a format that cannot run code when it loads: a numpy
``.npz`` for ``linear``, ``mlp`` and ``lstm``, XGBoost UBJ for ``xgboost`` and
LightGBM text for ``lightgbm``. A ``blend`` has no file of its own: it loads
the member versions its manifest names and rank-averages them. XGBoost,
LightGBM and polars are imported only when needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import narwhals as nw
import numpy as np


@dataclass(frozen=True)
class _Kind:
    files: Tuple[str, ...]
    load: Callable[[Dict[str, Path]], Any]
    predict: Callable[[Any, np.ndarray], np.ndarray]


def _load_linear(paths):
    with np.load(paths["model"], allow_pickle=False) as weights:
        return weights["coef"], weights["intercept"]


def _predict_linear(weights, X):
    coef, intercept = weights
    return X.astype(np.float64) @ coef.T + intercept


def _load_arrays(paths):
    with np.load(paths["model"], allow_pickle=False) as weights:
        return {key: weights[key] for key in weights.files}


def _predict_mlp(weights, X):
    h = (X.astype(np.float64) - weights["x_mean"]) / weights["x_scale"]
    layers = sum(key.startswith("w") for key in weights)
    for k in range(layers):
        h = h @ weights[f"w{k}"] + weights[f"b{k}"]
        if k < layers - 1:
            h = np.maximum(h, 0.0)
    return h


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _predict_lstm(weights, X):
    # The features are listed oldest lag first, one step of the sequence at a time.
    z = (X.astype(np.float64) - weights["x_mean"]) / weights["x_scale"]
    width, hidden = weights["w_input"].shape[0], weights["w_hidden"].shape[0]
    steps = z.reshape(len(z), -1, width)
    h = np.zeros((len(z), hidden))
    c = np.zeros_like(h)
    for t in range(steps.shape[1]):
        gates = (
            steps[:, t] @ weights["w_input"] + h @ weights["w_hidden"] + weights["b"]
        )
        i, f, g, o = np.split(gates, 4, axis=1)
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
    return h @ weights["w_out"] + weights["b_out"]


def _load_xgboost(paths):
    import xgboost

    booster = xgboost.Booster()
    booster.load_model(str(paths["model"]))
    return booster


def _predict_xgboost(booster, X):
    import xgboost

    return booster.predict(xgboost.DMatrix(X))


def _load_lightgbm(paths):
    import lightgbm

    return [
        lightgbm.Booster(model_file=str(paths[key]))
        for key in ("model_10d", "model_30d")
    ]


def _predict_lightgbm(boosters, X):
    return np.column_stack([booster.predict(X) for booster in boosters])


KINDS: Dict[str, _Kind] = {
    "linear": _Kind(("model",), _load_linear, _predict_linear),
    "mlp": _Kind(("model",), _load_arrays, _predict_mlp),
    "lstm": _Kind(("model",), _load_arrays, _predict_lstm),
    "xgboost": _Kind(("model",), _load_xgboost, _predict_xgboost),
    "lightgbm": _Kind(("model_10d", "model_30d"), _load_lightgbm, _predict_lightgbm),
}


def _ranks(raw: np.ndarray) -> np.ndarray:
    """Each column's average rank within the rows, scaled from 0 to 1."""
    import polars as pl

    return (
        pl.DataFrame(np.asarray(raw), orient="row")
        .select((pl.all().rank("average") - 1) / (pl.len() - 1))
        .fill_nan(0.5)
        .to_numpy()
    )


class ZooModel:
    """One loaded zoo model version."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        kind: Optional[_Kind],
        model: Any,
        client,
        members: Tuple[Tuple["ZooModel", float], ...] = (),
    ):
        self.manifest = manifest
        self._kind = kind
        self._model = model
        self._client = client
        self._members = members

    def __repr__(self) -> str:
        return f"ZooModel({self.manifest['ref']!r})"

    @property
    def features(self) -> List[str]:
        """The feature columns the model reads, in the order it reads them."""
        return list(self.manifest["features"])

    def predict(self, frame) -> "Any":
        """Predicts one inference release.

        Args:
            frame: A polars or pandas frame with an ``id`` column and the
                model's features.

        Returns:
            A polars frame with ``id``, ``pred_10d`` and ``pred_30d``. Each
            prediction is ranked within the frame and scaled from 0 to 1, the
            shape a submission needs.
        """
        import polars as pl

        data = nw.from_native(frame, eager_only=True)
        missing = [c for c in ["id", *self.features] if c not in data.columns]
        if missing:
            raise ValueError(
                f"{self.manifest['ref']} needs columns the frame lacks: {', '.join(missing)}."
            )
        raw = self._raw(data)
        outputs = self.manifest["outputs"]
        return pl.DataFrame(
            {"id": data["id"].to_list(), **dict(zip(outputs, raw.T))}
        ).with_columns(
            ((pl.col(c).rank("average") - 1) / (pl.len() - 1))
            .fill_nan(0.5)
            .fill_null(0.5)
            for c in outputs
        )

    def _raw(self, data) -> np.ndarray:
        if self._members:
            return sum(weight * _ranks(m._raw(data)) for m, weight in self._members)
        X = data.select(self.features).to_numpy().astype(np.float32)
        return np.asarray(self._kind.predict(self._model, X)).reshape(len(X), -1)

    def oof(self) -> "Any":
        """The out-of-fold predictions on the training data.

        Returns:
            A polars frame with ``id``, ``date``, ``pred_10d``, ``pred_30d``
            and ``fold`` (1 to 4 for development folds, 5 for the out-of-sample block).
        """
        import polars as pl

        path = self._client.download_model(self.manifest["ref"], "oof_csv")
        return pl.read_csv(path).with_columns(pl.col("date").str.to_date())


def load(ref: str, client: Optional[Any] = None) -> ZooModel:
    """Downloads a zoo model, checks its files and loads it.

    Args:
        ref: ``name@version``, or a bare name for the newest live version.
        client: A ``ChallengeClient``. Defaults to one that reads
            ``CROWDCENT_API_KEY`` from the environment.

    Returns:
        A ``ZooModel`` with ``.manifest``, ``.features``, ``.predict(frame)``
        and ``.oof()``.
    """
    if client is None:
        from .client import ChallengeClient

        client = ChallengeClient()
    manifest = client.get_model(ref)["manifest"]
    if manifest["kind"] == "blend":
        members = tuple(
            (load(member["ref"], client), float(member["weight"]))
            for member in manifest["members"]
        )
        return ZooModel(manifest, None, None, client, members)
    kind = KINDS.get(manifest["kind"])
    if kind is None:
        raise ValueError(
            f"{manifest['ref']} is a '{manifest['kind']}' model, which this client "
            f"cannot load. It loads: {', '.join(KINDS)} and blend."
        )
    paths = {key: client.download_model(manifest["ref"], key) for key in kind.files}
    return ZooModel(manifest, kind, kind.load(paths), client)
