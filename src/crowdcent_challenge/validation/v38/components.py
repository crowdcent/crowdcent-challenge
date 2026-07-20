"""Train component families as candidate-factory inputs — not slot recipes."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import numpy as np

from crowdcent_challenge.validation.training import (
    NotebookModelConfig,
    OptionalTrainingDependencyError,
    fit_sequence_model,
    fit_xgb_family,
    midpoint_rank,
    predict_xgb_family,
    recency_weights,
    refit_sequence_model,
    sequence_model_builders,
    tune_xgb_family,
)

from .contracts import PredictionPanel, PreparedDataset

logger = logging.getLogger(__name__)

COMPONENT_IDS: tuple[str, ...] = ("xgb", "lstm", "transformer", "bilstm")


class ComponentProvenance(str, Enum):
    REAL_MODEL = "real_model"
    PROXY_XGB = "proxy_xgb"


@dataclass(frozen=True)
class ComponentFitConfig:
    """How component prediction panels are produced for a fold."""

    mode: str = "real"
    calibration_days: int = 120
    embargo_days: int = 31
    notebook_config: NotebookModelConfig | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in {"real", "smoke"}:
            raise ValueError("component mode must be 'real' or 'smoke'")


@dataclass(frozen=True)
class ComponentBundle:
    panels: Mapping[str, PredictionPanel]
    provenance: Mapping[str, ComponentProvenance]


def fit_component_predictions(
    train: PreparedDataset,
    validation: PreparedDataset,
    *,
    config: ComponentFitConfig,
) -> ComponentBundle:
    if config.mode == "smoke":
        panels = _fit_smoke_proxies(train, validation, seed=config.seed)
        provenance = dict.fromkeys(panels, ComponentProvenance.PROXY_XGB)
        logger.info("component fit: smoke proxy XGB panels for %s", sorted(panels))
        return ComponentBundle(panels=panels, provenance=provenance)

    panels: dict[str, PredictionPanel] = {}
    provenance: dict[str, ComponentProvenance] = {}
    notebook_config = config.notebook_config or _default_research_notebook_config()

    panels["xgb"] = _fit_real_xgb_panel(
        train,
        validation,
        config=config,
        notebook_config=notebook_config,
    )
    provenance["xgb"] = ComponentProvenance.REAL_MODEL
    logger.info("component fit: real XGB panel on %d train / %d val rows", len(train.periods), len(validation.periods))

    sequence_panels = _fit_real_sequence_panels(
        train,
        validation,
        config=config,
        notebook_config=notebook_config,
    )
    if sequence_panels is not None:
        panels.update(sequence_panels)
        for component_id in ("lstm", "transformer", "bilstm"):
            provenance[component_id] = ComponentProvenance.REAL_MODEL
        logger.info("component fit: real LSTM/Transformer/BiLSTM panels")
    else:
        proxy_variants = {
            "lstm": config.seed + 1,
            "transformer": config.seed + 2,
            "bilstm": config.seed + 3,
        }
        for component_id, seed in proxy_variants.items():
            panels[component_id] = _fit_proxy_xgb_panel(
                train,
                validation,
                seed=seed,
                colsample_bytree=0.5 + 0.1 * (seed % 3),
            )
            provenance[component_id] = ComponentProvenance.PROXY_XGB
        logger.warning(
            "component fit: sequence models unavailable — %s use proxy XGB",
            sorted(proxy_variants),
        )
    return ComponentBundle(panels=panels, provenance=provenance)


def _default_research_notebook_config() -> NotebookModelConfig:
    return NotebookModelConfig(
        xgb_max_trees=150,
        xgb_min_trees=50,
        xgb_tree_step=25,
        xgb_feature_limit=250,
        xgb_seeds=(42, 1337),
    )


def _split_train_for_tuning(
    train: PreparedDataset,
    *,
    calibration_days: int,
    embargo_days: int,
) -> tuple[PreparedDataset, PreparedDataset]:
    unique = train.unique_periods
    if len(unique) < calibration_days + embargo_days + 30:
        split = max(1, int(len(unique) * 0.8))
        inner_dates = set(unique[:split])
        cal_dates = set(unique[split:])
    else:
        calibration_end = unique[-(embargo_days + 1)]
        calibration_start = calibration_end - timedelta(days=calibration_days - 1)
        inner_train_end = calibration_start - timedelta(days=embargo_days + 1)
        inner_dates = {period for period in unique if unique[0] <= period <= inner_train_end}
        cal_dates = {period for period in unique if calibration_start <= period <= calibration_end}
    inner_mask = np.asarray([period in inner_dates for period in train.periods])
    cal_mask = np.asarray([period in cal_dates for period in train.periods])
    if not inner_mask.any() or not cal_mask.any():
        split = max(1, int(len(unique) * 0.8))
        inner_dates = set(unique[:split])
        cal_dates = set(unique[split:])
        inner_mask = np.asarray([period in inner_dates for period in train.periods])
        cal_mask = np.asarray([period in cal_dates for period in train.periods])
    return train.take(inner_mask), train.take(cal_mask)


def _fit_real_xgb_panel(
    train: PreparedDataset,
    validation: PreparedDataset,
    *,
    config: ComponentFitConfig,
    notebook_config: NotebookModelConfig,
) -> PredictionPanel:
    try:
        inner_train, calibration = _split_train_for_tuning(
            train,
            calibration_days=config.calibration_days,
            embargo_days=config.embargo_days,
        )
        tuning = tune_xgb_family(
            train_features=inner_train.features,
            train_targets=inner_train.targets,
            train_weights=recency_weights(inner_train.periods),
            calibration_features=calibration.features,
            calibration_targets=calibration.targets,
            calibration_dates=calibration.periods,
            config=notebook_config,
        )
        bags = fit_xgb_family(
            features=train.features,
            targets=train.targets,
            sample_weights=recency_weights(train.periods),
            tuning=tuning,
            config=notebook_config,
        )
        predictions = predict_xgb_family(bags, validation.features, tuning)
    except OptionalTrainingDependencyError as exc:
        logger.warning("real XGB unavailable (%s); falling back to proxy panel", exc)
        return _fit_proxy_xgb_panel(train, validation, seed=config.seed, colsample_bytree=0.8)
    return PredictionPanel(
        periods=validation.periods,
        ids=validation.ids,
        values=predictions,
    )


def _fit_real_sequence_panels(
    train: PreparedDataset,
    validation: PreparedDataset,
    *,
    config: ComponentFitConfig,
    notebook_config: NotebookModelConfig,
) -> dict[str, PredictionPanel] | None:
    if train.sequence_features is None or train.sequence_schema is None:
        return None
    try:
        builders = sequence_model_builders(
            lag_windows=train.sequence_schema.lag_windows,
            n_features_per_timestep=train.sequence_schema.n_features_per_timestep,
        )
        inner_train, calibration = _split_train_for_tuning(
            train,
            calibration_days=config.calibration_days,
            embargo_days=config.embargo_days,
        )
        panels: dict[str, PredictionPanel] = {}
        for offset, component_id in enumerate(("lstm", "transformer", "bilstm")):
            fit = fit_sequence_model(
                builders[component_id],
                train_features=inner_train.sequence_features,
                train_targets=inner_train.targets,
                calibration_features=calibration.sequence_features,
                calibration_targets=calibration.targets,
                config=notebook_config,
            )
            model = refit_sequence_model(
                builders[component_id],
                features=train.sequence_features,
                targets=train.targets,
                epochs=fit.best_epoch,
                config=notebook_config,
                seed=config.seed + 1 + offset,
            )
            raw = np.asarray(model.predict(validation.sequence_features, verbose=0), dtype=float)
            values = np.column_stack([midpoint_rank(raw[:, horizon]) for horizon in range(2)])
            panels[component_id] = PredictionPanel(
                periods=validation.periods,
                ids=validation.ids,
                values=values,
            )
        return panels
    except OptionalTrainingDependencyError as exc:
        logger.warning("real sequence models unavailable (%s); using proxy XGB", exc)
        return None


def _fit_smoke_proxies(
    train: PreparedDataset,
    validation: PreparedDataset,
    *,
    seed: int,
) -> dict[str, PredictionPanel]:
    variant_params = {
        "xgb": {"seed": seed, "colsample_bytree": 0.8},
        "lstm": {"seed": seed + 1, "colsample_bytree": 0.6},
        "transformer": {"seed": seed + 2, "colsample_bytree": 0.7},
        "bilstm": {"seed": seed + 3, "colsample_bytree": 0.5},
    }
    return {
        name: _fit_proxy_xgb_panel(train, validation, **params)
        for name, params in variant_params.items()
    }


def _fit_proxy_xgb_panel(
    train: PreparedDataset,
    validation: PreparedDataset,
    *,
    seed: int,
    colsample_bytree: float,
) -> PredictionPanel:
    xgboost = _import_xgboost()
    predictions = np.empty((len(validation.periods), 2), dtype=float)
    for horizon in range(2):
        model = xgboost.XGBRegressor(
            n_estimators=30,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(train.features, train.targets[:, horizon], verbose=False)
        predictions[:, horizon] = model.predict(validation.features)
    return PredictionPanel(
        periods=validation.periods,
        ids=validation.ids,
        values=predictions,
    )


def _import_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError(
            "v3.8 components require xgboost; install the validation-training extra"
        ) from exc
    return xgboost
