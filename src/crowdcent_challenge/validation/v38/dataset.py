"""Load Hyperliquid panels into the v3.8 runner contracts."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from crowdcent_challenge.validation.pit import MetaPurpose, latest_available_meta_release
from crowdcent_challenge.validation.training import sequence_feature_schema, sequence_matrix
from crowdcent_challenge.validation.training.features import FEATURE_RE

from .contracts import MetaInputPanel, PredictionPanel, PreparedDataset

META_COLUMNS = {"id", "release_date", "pred_10d", "pred_30d"}
TARGET_COLUMNS = ("target_10d", "target_30d")
DEFAULT_META_PATHS: tuple[Path, ...] = (
    Path("/var/lib/centaur/gdrive/cc-main/Slides/Visuals/data/meta_model.parquet"),
    Path("/var/lib/centaur/workspace/.tmp/usebt/data/meta_model.parquet"),
    Path("/var/lib/centaur/workspace/.tmp/btwork/meta_model.parquet"),
)


def load_meta_frame(
    *,
    preferred_paths: Sequence[str | Path] | None = None,
    allow_download: bool = False,
):
    """Resolve a meta-model parquet from local paths before optional API download."""

    pl = _import_polars()
    candidates = [Path(path) for path in (preferred_paths or DEFAULT_META_PATHS)]
    for path in candidates:
        if path.exists():
            return pl.read_parquet(path).with_columns(pl.col("release_date").cast(pl.Date))
    if allow_download:
        import crowdcent_challenge as cc

        client = cc.ChallengeClient(challenge_slug="hyperliquid-ranking")
        download_path = candidates[0]
        download_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_meta_model(dest_path=str(download_path))
        return pl.read_parquet(download_path).with_columns(
            pl.col("release_date").cast(pl.Date)
        )
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"meta_model.parquet not found locally (searched: {searched}); "
        "place it under Visuals/data/ or pass allow_download=True"
    )


def prepared_dataset_from_frame(frame) -> PreparedDataset:
    """Convert a merged CrowdCent training frame into a row-aligned dataset."""

    pl = _import_polars()
    data = frame.with_columns(pl.col("date").cast(pl.Date))
    data = data.filter(
        pl.all_horizontal([pl.col(target).is_not_null() for target in TARGET_COLUMNS])
    )
    meta = {"id", "eodhd_id", "date", *TARGET_COLUMNS}
    feature_columns = [
        column
        for column in data.columns
        if column not in meta and data[column].dtype in (pl.Float32, pl.Float64, pl.Int64)
    ]
    if not feature_columns:
        raise ValueError("no numeric feature columns found in training frame")
    matrix = (
        data.select(feature_columns)
        .fill_null(0.0)
        .fill_nan(0.0)
        .to_numpy()
        .astype(np.float32)
    )
    lag_columns = [column for column in data.columns if FEATURE_RE.match(column)]
    sequence_features = None
    sequence_schema = None
    if lag_columns:
        sequence_schema = sequence_feature_schema(lag_columns)
        sequence_features = sequence_matrix(data, sequence_schema)
    return PreparedDataset(
        periods=tuple(data["date"].to_list()),
        ids=tuple(data["id"].to_list()),
        targets=data.select(TARGET_COLUMNS).to_numpy().astype(float),
        features=matrix,
        sequence_features=sequence_features,
        sequence_schema=sequence_schema,
    )


def build_meta_panels_for_dataset(
    dataset: PreparedDataset,
    meta_frame,
) -> tuple[dict[date, PredictionPanel], dict[date, PredictionPanel]]:
    """Build same-period scoring meta and lagged model-input meta panels."""

    pl = _import_polars()
    meta = meta_frame.with_columns(pl.col("release_date").cast(pl.Date))
    release_dates = tuple(sorted(meta["release_date"].unique().to_list()))
    score_meta: dict[date, PredictionPanel] = {}
    input_meta: dict[date, PredictionPanel] = {}
    for period in dataset.unique_periods:
        mask = np.asarray([row_period == period for row_period in dataset.periods])
        ids = tuple(dataset.ids[index] for index in np.flatnonzero(mask))
        if len(ids) < 10:
            continue
        score_panel = _meta_panel_for_release(meta, period, ids)
        if score_panel is None:
            continue
        score_meta[period] = score_panel
        prior = latest_available_meta_release(
            release_dates,
            period,
            MetaPurpose.MODEL_INPUT,
        )
        if prior is None:
            values = np.full((len(ids), 2), 0.5, dtype=float)
            input_meta[period] = PredictionPanel(
                periods=tuple(period for _ in ids),
                ids=ids,
                values=values,
            )
            continue
        prior_panel = _meta_panel_for_release(meta, prior, ids)
        if prior_panel is None:
            values = np.full((len(ids), 2), 0.5, dtype=float)
        else:
            values = prior_panel.values
        input_meta[period] = PredictionPanel(
            periods=tuple(period for _ in ids),
            ids=ids,
            values=values,
        )
    return score_meta, input_meta


def _meta_panel_for_release(meta_frame, release_date: date, ids: Iterable[str]):
    pl = _import_polars()
    subset = meta_frame.filter(pl.col("release_date") == release_date)
    if not subset.height:
        return None
    lookup = {
        row["id"]: (row["pred_10d"], row["pred_30d"])
        for row in subset.select(["id", "pred_10d", "pred_30d"]).iter_rows(named=True)
    }
    values = []
    ordered_ids = tuple(ids)
    for asset_id in ordered_ids:
        prediction = lookup.get(asset_id)
        if prediction is None:
            values.append((0.5, 0.5))
        else:
            values.append(prediction)
    return PredictionPanel(
        periods=tuple(release_date for _ in ordered_ids),
        ids=ordered_ids,
        values=np.asarray(values, dtype=float),
    )


def meta_input_panel_from_store(
    reference: PredictionPanel,
    input_meta: Mapping[date, PredictionPanel],
) -> MetaInputPanel:
    release_dates: list[date] = []
    values: list[np.ndarray] = []
    for period, asset_id in zip(reference.periods, reference.ids, strict=True):
        panel = input_meta[period]
        lookup = dict(zip(panel.ids, panel.values, strict=True))
        release = latest_available_meta_release(
            input_meta,
            period,
            MetaPurpose.MODEL_INPUT,
        )
        release_dates.append(release or period - timedelta(days=1))
        values.append(lookup[asset_id])
    return MetaInputPanel(
        predictions=PredictionPanel(
            periods=reference.periods,
            ids=reference.ids,
            values=np.vstack(values),
        ),
        release_dates=tuple(release_dates),
    )


def _import_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "PreparedDataset loading requires polars; install validation-training extra"
        ) from exc
    return pl
