"""Full-fidelity notebook model families behind optional dependencies."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from crowdcent_challenge.scoring import spearman_correlation

from ._optional import OptionalTrainingDependencyError, import_training_dependency
from .recipe import midpoint_rank


@dataclass(frozen=True)
class NotebookModelConfig:
    """Production defaults from Hyperliquid Ranking v3.6."""

    xgb_max_trees: int = 600
    xgb_importance_trees: int = 300
    xgb_feature_limit: int = 500
    xgb_min_trees: int = 100
    xgb_tree_step: int = 25
    xgb_device: str = "cpu"
    recency_halflife_days: int = 365
    xgb_seeds: tuple[int, ...] = (42, 1337, 2024, 7, 3407)
    transformer_seeds: tuple[int, ...] = (42, 1337)
    sequence_epochs: int = 20
    sequence_batch_size: int = 1024
    early_stopping_patience: int = 3


def recency_weights(
    dates: Iterable[date],
    *,
    halflife_days: int = 365,
) -> np.ndarray:
    """Exponential sample weights relative to the fold's own training end."""

    if halflife_days <= 0:
        raise ValueError("halflife_days must be positive")
    values = np.asarray(list(dates), dtype="datetime64[D]")
    if not len(values):
        raise ValueError("dates cannot be empty")
    age_days = (values.max() - values).astype(int)
    return (0.5 ** (age_days / halflife_days)).astype(np.float32)


def mean_daily_spearman(
    dates: Iterable[date],
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    min_assets: int = 10,
) -> float:
    """Cross-sectional Spearman averaged over eligible dates."""

    date_values = np.asarray(list(dates))
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if not (len(date_values) == len(targets) == len(predictions)):
        raise ValueError("dates, targets, and predictions must align")
    scores = []
    for period in np.unique(date_values):
        mask = date_values == period
        if mask.sum() >= min_assets:
            scores.append(spearman_correlation(targets[mask], predictions[mask]))
    return float(np.mean(scores)) if scores else float("nan")


@dataclass(frozen=True)
class XGBTuning:
    feature_indices: np.ndarray
    tree_counts: tuple[int, int]


def _xgb_params(config: NotebookModelConfig) -> dict[str, Any]:
    return {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": config.xgb_device,
        "n_jobs": -1,
    }


def tune_xgb_family(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_dates: Iterable[date],
    config: NotebookModelConfig | None = None,
) -> XGBTuning:
    """Select the notebook's feature union and per-horizon tree budgets."""

    config = config or NotebookModelConfig()
    xgboost = import_training_dependency("xgboost")
    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    x_cal = np.asarray(calibration_features, dtype=np.float32)
    y_cal = np.asarray(calibration_targets, dtype=np.float32)
    weights = np.asarray(train_weights, dtype=np.float32)
    if y_train.ndim != 2 or y_train.shape[1] != 2:
        raise ValueError("train_targets must have shape (rows, 2)")
    if y_cal.ndim != 2 or y_cal.shape[1] != 2:
        raise ValueError("calibration_targets must have shape (rows, 2)")
    if not (len(x_train) == len(y_train) == len(weights)):
        raise ValueError("training arrays must align")
    if len(x_cal) != len(y_cal):
        raise ValueError("calibration arrays must align")

    importance = np.zeros(x_train.shape[1], dtype=float)
    selector_params = {
        "n_estimators": config.xgb_importance_trees,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": config.xgb_device,
        "random_state": 42,
        "n_jobs": -1,
    }
    for horizon in range(2):
        selector = xgboost.XGBRegressor(**selector_params)
        selector.fit(x_train, y_train[:, horizon], verbose=False)
        importance += selector.feature_importances_
    nonzero = int((importance > 0).sum())
    feature_count = min(
        config.xgb_feature_limit,
        nonzero or x_train.shape[1],
    )
    feature_indices = np.sort(np.argsort(importance)[::-1][:feature_count])

    selected_train = np.ascontiguousarray(x_train[:, feature_indices])
    selected_cal = np.ascontiguousarray(x_cal[:, feature_indices])
    grid = list(
        range(
            max(1, min(50, config.xgb_max_trees)),
            config.xgb_max_trees + 1,
            config.xgb_tree_step,
        )
    )
    if config.xgb_max_trees not in grid:
        grid.append(config.xgb_max_trees)
    tree_counts = []
    for horizon in range(2):
        staged = xgboost.XGBRegressor(
            n_estimators=config.xgb_max_trees,
            random_state=42,
            **_xgb_params(config),
        )
        staged.fit(
            selected_train,
            y_train[:, horizon],
            sample_weight=weights,
            verbose=False,
        )
        scored = [
            (
                mean_daily_spearman(
                    calibration_dates,
                    y_cal[:, horizon],
                    staged.predict(selected_cal, iteration_range=(0, trees)),
                ),
                trees,
            )
            for trees in grid
        ]
        best_trees = max(scored, key=lambda item: item[0])[1]
        tree_counts.append(min(config.xgb_max_trees, max(config.xgb_min_trees, best_trees)))
    return XGBTuning(
        feature_indices=feature_indices,
        tree_counts=(tree_counts[0], tree_counts[1]),
    )


def fit_xgb_family(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    tuning: XGBTuning,
    config: NotebookModelConfig | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Refit the five-seed, two-horizon XGB family on an outer train fold."""

    config = config or NotebookModelConfig()
    xgboost = import_training_dependency("xgboost")
    selected = np.ascontiguousarray(
        np.asarray(features, dtype=np.float32)[:, tuning.feature_indices]
    )
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    bags = []
    for horizon in range(2):
        models = []
        for seed in config.xgb_seeds:
            model = xgboost.XGBRegressor(
                n_estimators=tuning.tree_counts[horizon],
                random_state=seed,
                **_xgb_params(config),
            )
            model.fit(
                selected,
                targets[:, horizon],
                sample_weight=weights,
                verbose=False,
            )
            models.append(model)
        bags.append(tuple(models))
    return bags[0], bags[1]


def predict_xgb_family(
    bags: tuple[tuple[Any, ...], tuple[Any, ...]],
    features: np.ndarray,
    tuning: XGBTuning,
) -> np.ndarray:
    """Predict a two-horizon XGB seed bag, averaging in rank space."""

    selected = np.ascontiguousarray(
        np.asarray(features, dtype=np.float32)[:, tuning.feature_indices]
    )
    horizons = [
        midpoint_rank(
            np.mean(
                [midpoint_rank(model.predict(selected)) for model in horizon_bag],
                axis=0,
            )
        )
        for horizon_bag in bags
    ]
    return np.column_stack(horizons)


def _sequence_imports() -> tuple[Any, Any, Any, Any, Any]:
    keras = import_training_dependency("keras")
    backend = keras.backend.backend()
    if backend != "jax":
        raise OptionalTrainingDependencyError(
            "Full-fidelity sequence validation requires the JAX Keras backend. "
            "Set KERAS_BACKEND=jax before Python starts."
        )
    estimators = import_training_dependency("centimators.model_estimators")
    losses = import_training_dependency("centimators.losses")
    return (
        keras,
        estimators.LSTMRegressor,
        estimators.TransformerRegressor,
        losses.SpearmanCorrelation,
        losses.CombinedLoss,
    )


def sequence_model_builders(
    *,
    lag_windows: tuple[int, ...],
    n_features_per_timestep: int,
) -> dict[str, Callable[[], Any]]:
    """Return exact v3.6 LSTM, Transformer, and BiLSTM constructors."""

    (
        _keras,
        lstm_regressor,
        transformer_regressor,
        spearman_loss,
        combined_loss,
    ) = _sequence_imports()

    def build_lstm() -> Any:
        return lstm_regressor(
            output_units=2,
            lag_windows=list(lag_windows),
            n_features_per_timestep=n_features_per_timestep,
            lstm_units=[(64, 0.1, 0.0), (32, 0.1, 0.0)],
            use_layer_norm=True,
            loss_function=combined_loss(mse_weight=1.0, spearman_weight=1.0),
        )

    def build_transformer() -> Any:
        return transformer_regressor(
            output_units=2,
            lag_windows=list(lag_windows),
            n_features_per_timestep=n_features_per_timestep,
            d_model=48,
            num_heads=4,
            num_blocks=2,
            ff_dim=96,
            dropout_rate=0.1,
            attention_type="temporal",
            pooling_type="attention",
            loss_function=combined_loss(mse_weight=0.5, spearman_weight=2.0),
        )

    def build_bilstm() -> Any:
        return lstm_regressor(
            output_units=2,
            lag_windows=list(lag_windows),
            n_features_per_timestep=n_features_per_timestep,
            lstm_units=[(48, 0.1, 0.0)],
            bidirectional=True,
            use_layer_norm=True,
            loss_function=spearman_loss(),
        )

    return {
        "lstm": build_lstm,
        "transformer": build_transformer,
        "bilstm": build_bilstm,
    }


@dataclass(frozen=True)
class SequenceFit:
    model: Any
    best_epoch: int
    calibration_predictions: np.ndarray


def fit_sequence_model(
    builder: Callable[[], Any],
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    config: NotebookModelConfig | None = None,
) -> SequenceFit:
    """Tune one sequence model with the notebook's early-stopping policy."""

    config = config or NotebookModelConfig()
    keras, *_ = _sequence_imports()
    model = builder()
    model.fit(
        train_features,
        train_targets,
        epochs=config.sequence_epochs,
        batch_size=config.sequence_batch_size,
        validation_data=(calibration_features, calibration_targets),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    history = getattr(model.model, "history", None)
    val_losses = (history.history.get("val_loss", []) if history else []) or []
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else config.sequence_epochs
    predictions = np.asarray(model.predict(calibration_features, verbose=0), dtype=float)
    return SequenceFit(
        model=model,
        best_epoch=best_epoch,
        calibration_predictions=predictions,
    )


def refit_sequence_model(
    builder: Callable[[], Any],
    *,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    config: NotebookModelConfig | None = None,
    seed: int | None = None,
) -> Any:
    """Refit a fresh sequence model on an outer training fold."""

    config = config or NotebookModelConfig()
    keras, *_ = _sequence_imports()
    if seed is not None:
        keras.utils.set_random_seed(seed)
    model = builder()
    model.fit(
        features,
        targets,
        epochs=epochs,
        batch_size=config.sequence_batch_size,
        verbose=2,
    )
    return model
