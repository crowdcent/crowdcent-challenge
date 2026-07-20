from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from crowdcent_challenge.validation.training import (
    HorizonRecipe,
    NestedCalibrationConfig,
    NotebookModelConfig,
    RawHorizonPredictions,
    StackWeights,
    build_five_slot_predictions,
    build_nested_walk_forward_folds,
    engineer_cross_sectional_ranks,
    engineer_lag_differences,
    fit_xgb_family,
    mean_daily_spearman,
    midpoint_rank,
    predict_xgb_family,
    recency_weights,
    sequence_feature_schema,
    sequence_matrix,
    tune_xgb_family,
    validate_prepared_frame,
)
from crowdcent_challenge.validation.walkforward import WalkForwardConfig


def _feature_columns(feature_count=2):
    return [
        f"feature_{feature}_lag{lag}" for feature in range(feature_count) for lag in (0, 5, 10, 15)
    ]


def test_sequence_schema_is_lag_major_and_rejects_missing_lags():
    columns = _feature_columns()
    schema = sequence_feature_schema(reversed(columns))

    assert schema.ordered_columns[:2] == ("feature_0_lag0", "feature_1_lag0")
    assert schema.ordered_columns[-2:] == ("feature_0_lag15", "feature_1_lag15")
    assert schema.n_features_per_timestep == 2

    with pytest.raises(ValueError, match="incomplete sequence lag grid"):
        sequence_feature_schema(columns[:-1])


def test_feature_engineering_and_prepared_frame_validation():
    columns = _feature_columns(feature_count=1)
    schema = sequence_feature_schema(columns)
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "id": ["BTC", "ETH"],
            "target_10d": [0.0, 1.0],
            "target_30d": [1.0, 0.0],
            "feature_0_lag0": [0.8, 0.2],
            "feature_0_lag5": [0.6, 0.3],
            "feature_0_lag10": [0.4, 0.4],
            "feature_0_lag15": [0.2, 0.5],
        }
    )
    validate_prepared_frame(frame)
    with_diffs, diff_columns = engineer_lag_differences(frame, schema)
    with_ranks, rank_columns = engineer_cross_sectional_ranks(
        with_diffs,
        diff_columns,
    )

    assert len(diff_columns) == 4
    assert len(rank_columns) == 4
    assert set(rank_columns).issubset(with_ranks.columns)
    assert sequence_matrix(frame, schema).shape == (2, 4)

    duplicate = pl.concat([frame, frame[:1]])
    with pytest.raises(ValueError, match="duplicate point-in-time keys"):
        validate_prepared_frame(duplicate)


def test_notebook_recipe_builds_five_aligned_ranked_slots():
    n_assets = 20
    raw = RawHorizonPredictions(
        xgb=np.linspace(0.0, 1.0, n_assets),
        lstm=np.linspace(1.0, 0.0, n_assets),
        transformer=np.sin(np.linspace(0.0, 4.0, n_assets)),
        bilstm=np.cos(np.linspace(0.0, 4.0, n_assets)),
        meta=np.linspace(0.2, 0.8, n_assets) ** 2,
    )
    recipe = HorizonRecipe(
        slot1_meta_weight=0.4,
        slot2_meta_weight=0.25,
        slot3_slot1_weight=0.35,
        slot4_slot1_weight=0.5,
        slot4_meta_weight=0.25,
        slot5=StackWeights(
            xgb=0.2,
            lstm=0.15,
            transformer=0.25,
            bilstm=0.15,
            meta=0.25,
        ),
    )

    slots = build_five_slot_predictions(raw, recipe).as_dict()

    assert set(slots) == {1, 2, 3, 4, 5}
    assert all(values.shape == (n_assets,) for values in slots.values())
    assert all(0.0 < values.min() < values.max() < 1.0 for values in slots.values())
    assert midpoint_rank(np.arange(4)).tolist() == [0.125, 0.375, 0.625, 0.875]


def test_recency_and_daily_spearman_are_fold_local():
    periods = [
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]
    weights = recency_weights(periods, halflife_days=1)
    assert weights.tolist() == pytest.approx([0.25, 0.5, 1.0])

    repeated_dates = [date(2025, 1, 1)] * 10 + [date(2025, 1, 2)] * 10
    targets = np.tile(np.arange(10), 2)
    predictions = targets.copy()
    assert mean_daily_spearman(
        repeated_dates,
        targets,
        predictions,
    ) == pytest.approx(1.0)


def test_tiny_xgb_family_runs_end_to_end():
    rng = np.random.default_rng(7)
    train_rows = 60
    calibration_rows = 20
    x_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    x_calibration = rng.normal(size=(calibration_rows, 4)).astype(np.float32)
    y_train = np.column_stack(
        [
            midpoint_rank(x_train[:, 0] + rng.normal(0, 0.1, train_rows)),
            midpoint_rank(x_train[:, 1] + rng.normal(0, 0.1, train_rows)),
        ]
    )
    y_calibration = np.column_stack(
        [
            midpoint_rank(x_calibration[:, 0] + rng.normal(0, 0.1, calibration_rows)),
            midpoint_rank(x_calibration[:, 1] + rng.normal(0, 0.1, calibration_rows)),
        ]
    )
    calibration_dates = [
        date(2025, 1, 1) + timedelta(days=index // 10) for index in range(calibration_rows)
    ]
    config = NotebookModelConfig(
        xgb_max_trees=2,
        xgb_importance_trees=2,
        xgb_feature_limit=2,
        xgb_min_trees=1,
        xgb_tree_step=1,
        xgb_seeds=(42,),
    )
    tuning = tune_xgb_family(
        train_features=x_train,
        train_targets=y_train,
        train_weights=np.ones(train_rows),
        calibration_features=x_calibration,
        calibration_targets=y_calibration,
        calibration_dates=calibration_dates,
        config=config,
    )
    bags = fit_xgb_family(
        features=x_train,
        targets=y_train,
        sample_weights=np.ones(train_rows),
        tuning=tuning,
        config=config,
    )

    predictions = predict_xgb_family(bags, x_calibration, tuning)

    assert tuning.tree_counts == (2, 2)
    assert predictions.shape == (calibration_rows, 2)
    assert np.isfinite(predictions).all()


def test_nested_walk_forward_preserves_both_embargoes():
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(900)]
    folds = build_nested_walk_forward_folds(
        dates,
        outer_config=WalkForwardConfig(
            min_train_days=400,
            validation_days=60,
            embargo_days=30,
            max_folds=2,
        ),
        calibration_config=NestedCalibrationConfig(
            calibration_days=60,
            embargo_days=30,
            min_inner_train_days=200,
        ),
    )

    assert len(folds) == 2
    for fold in folds:
        assert fold.outer.embargo_days == 30
        assert fold.inner_embargo_days == 30
        assert fold.calibration_end <= fold.outer.train_end
        assert fold.calibration_end < fold.outer.validation_start
