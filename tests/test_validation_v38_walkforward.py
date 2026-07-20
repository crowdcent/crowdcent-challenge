"""Historical walk-forward backtests under counterfactual USE50."""

import pytest
from crowdcent_challenge.validation.v38 import (
    ComponentFitConfig,
    V38NestedFoldConfig,
    V38RunConfig,
    candidate_library,
    make_synthetic_dataset,
    run_v38_walkforward_backtest,
)


def test_walkforward_backtest_runs_counterfactual_use50_on_meta_era():
    dataset = make_synthetic_dataset(
        V38RunConfig(assets_per_period=20, n_features=8, seed=7)
    )
    result = run_v38_walkforward_backtest(
        dataset,
        config=V38RunConfig(
            assets_per_period=20,
            n_features=8,
            seed=7,
            fold_config=V38NestedFoldConfig(
                outer_min_train_days=360,
                outer_validation_days=60,
                outer_step_days=60,
                outer_max_folds=1,
                inner_min_train_days=180,
                inner_validation_days=45,
                inner_step_days=45,
                inner_folds=1,
                embargo_days=30,
            ),
            component_fit=ComponentFitConfig(mode="smoke", seed=7),
        ),
        specs=candidate_library(smoke=True),
        max_folds=1,
    )
    assert result.counterfactual_use50
    assert len(result.folds) == 1
    assert result.folds[0].fold_index == 0
    assert result.folds[0].result.outer_delta > 0.0


def test_walkforward_rejects_unavailable_fold_index():
    dataset = make_synthetic_dataset(
        V38RunConfig(assets_per_period=20, n_features=8, seed=7)
    )
    with pytest.raises(ValueError, match="outside available range"):
        run_v38_walkforward_backtest(
            dataset,
            config=V38RunConfig(
                assets_per_period=20,
                n_features=8,
                seed=7,
                fold_config=V38NestedFoldConfig(
                    outer_min_train_days=360,
                    outer_validation_days=60,
                    outer_step_days=60,
                    outer_max_folds=1,
                    inner_min_train_days=180,
                    inner_validation_days=45,
                    inner_step_days=45,
                    inner_folds=1,
                    embargo_days=30,
                ),
                component_fit=ComponentFitConfig(mode="smoke", seed=7),
            ),
            specs=candidate_library(smoke=True),
            max_folds=1,
            fold_indices=(1,),
        )
