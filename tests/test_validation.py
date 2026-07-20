from datetime import date, timedelta

import numpy as np
import pytest
from crowdcent_challenge.validation import (
    HYPERLIQUID_RANKING_ERAS,
    AtomicPeriodScore,
    InsufficientAssetsError,
    LiveSlotPercentile,
    MetaPurpose,
    PointsObjective,
    PromotionPolicy,
    ShadowPercentile,
    WalkForwardConfig,
    book_objective,
    build_walk_forward_folds,
    evaluate_scored_subsets,
    evaluate_shadow_promotion,
    latest_available_meta_release,
    meta_coverage,
    score_atomic_period,
    self_inclusion_diagnostic,
    universe_churn,
    weighted_meta_model,
)


def test_challenge_eras_keep_meta_and_points_cutovers_separate():
    assert not HYPERLIQUID_RANKING_ERAS.has_meta_model(date(2025, 6, 4))
    assert HYPERLIQUID_RANKING_ERAS.has_meta_model(date(2025, 6, 5))
    assert HYPERLIQUID_RANKING_ERAS.points_objective(date(2026, 6, 30)) is PointsObjective.RAW_ONLY
    assert (
        HYPERLIQUID_RANKING_ERAS.points_objective(date(2026, 7, 1))
        is PointsObjective.RAW_UNIQUE_50_50
    )
    assert HYPERLIQUID_RANKING_ERAS.performance_award_date(date(2026, 7, 1)) == date(2026, 8, 1)
    assert not HYPERLIQUID_RANKING_ERAS.is_fully_resolved(
        date(2026, 7, 1),
        date(2026, 7, 31),
    )


def test_meta_selection_distinguishes_model_input_from_scoring():
    releases = [date(2025, 6, 5), date(2025, 6, 6), date(2025, 6, 7)]
    period = date(2025, 6, 7)

    assert latest_available_meta_release(releases, period, MetaPurpose.MODEL_INPUT) == date(
        2025, 6, 6
    )
    assert (
        latest_available_meta_release(
            releases,
            period,
            MetaPurpose.UNIQUENESS_SCORING,
        )
        == period
    )
    assert (
        meta_coverage(
            [date(2025, 6, 5), date(2025, 6, 6)],
            releases,
            MetaPurpose.MODEL_INPUT,
        )
        == 0.5
    )


def test_atomic_score_exposes_eight_metrics_and_use50_identity():
    rng = np.random.default_rng(42)
    n_assets = 80
    true_10d = np.linspace(0.0, 1.0, n_assets)
    true_30d = np.linspace(1.0, 0.0, n_assets)
    pred_10d = true_10d + rng.normal(0.0, 0.1, n_assets)
    pred_30d = true_30d + rng.normal(0.0, 0.1, n_assets)
    meta_10d = 0.7 * true_10d + rng.normal(0.0, 0.2, n_assets)
    meta_30d = 0.7 * true_30d + rng.normal(0.0, 0.2, n_assets)

    score = score_atomic_period(
        y_true_10d=true_10d,
        y_pred_10d=pred_10d,
        y_true_30d=true_30d,
        y_pred_30d=pred_30d,
        meta_pred_10d=meta_10d,
        meta_pred_30d=meta_30d,
    )

    assert set(score.raw) == {
        "spearman_10d",
        "spearman_30d",
        "ndcg@40_10d",
        "ndcg@40_30d",
    }
    assert score.unique is not None
    assert len(score.unique) == 6
    assert score.use50_proxy == pytest.approx(
        0.5 * score.raw_composite + 0.5 * score.unique_composite
    )
    assert score.objective_proxy(PointsObjective.RAW_ONLY) == score.raw_composite


def test_atomic_score_requires_enough_assets_and_both_meta_horizons():
    values = np.linspace(0.0, 1.0, 9)
    with pytest.raises(InsufficientAssetsError):
        score_atomic_period(
            y_true_10d=values,
            y_pred_10d=values,
            y_true_30d=values,
            y_pred_30d=values,
        )

    values = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="both meta horizons"):
        score_atomic_period(
            y_true_10d=values,
            y_pred_10d=values,
            y_true_30d=values,
            y_pred_30d=values,
            meta_pred_10d=values,
        )


def test_book_objective_averages_only_selected_slots():
    raw = {
        "spearman_10d": 0.1,
        "spearman_30d": 0.2,
        "ndcg@40_10d": 0.6,
        "ndcg@40_30d": 0.7,
    }
    scores = {
        1: AtomicPeriodScore(raw=raw),
        2: AtomicPeriodScore(raw={name: value + 0.1 for name, value in raw.items()}),
    }

    assert book_objective(
        scores,
        PointsObjective.RAW_ONLY,
        scored_slots={2},
    ) == pytest.approx(scores[2].raw_composite)


def test_walk_forward_folds_enforce_calendar_embargo():
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(1_500)]
    config = WalkForwardConfig(
        min_train_days=365,
        validation_days=90,
        embargo_days=30,
        max_folds=3,
    )

    folds = build_walk_forward_folds(dates, config)

    assert len(folds) == 3
    assert all(fold.embargo_days == 30 for fold in folds)
    assert all(fold.train_end < fold.validation_start for fold in folds)
    assert folds[0].validation_start < folds[1].validation_start


def test_promotion_uses_only_resolved_matched_periods():
    start = date(2026, 8, 1)
    observations = [
        ShadowPercentile(
            period=start + timedelta(days=offset),
            raw_percentile=75.0,
            unique_percentile=65.0,
            resolved=offset != 2,
        )
        for offset in range(4)
    ]
    incumbent = {start + timedelta(days=offset): 60.0 for offset in range(4)}

    decision = evaluate_shadow_promotion(
        observations,
        incumbent,
        PromotionPolicy(
            min_resolved_periods=3,
            min_mean_margin=5.0,
            max_underperformance_frequency=0.25,
        ),
    )

    assert decision.promote
    assert decision.matched_periods == 3
    assert decision.candidate_mean == 70.0


def test_scored_subset_analysis_uses_common_resolved_periods():
    start = date(2026, 8, 1)
    observations = []
    for offset in range(21):
        period = start + timedelta(days=offset)
        observations.extend(
            [
                LiveSlotPercentile(period, 1, 75.0, 65.0, resolved=True),
                LiveSlotPercentile(period, 2, 65.0, 55.0, resolved=True),
            ]
        )
    observations.append(LiveSlotPercentile(date(2026, 8, 22), 2, 100.0, 100.0, resolved=False))

    results = evaluate_scored_subsets(observations)

    assert results[0].slots == (1,)
    assert results[0].periods == 21
    assert results[0].mean_percentile == 70.0
    assert {result.slots for result in results} == {(1,), (2,), (1, 2)}


def test_universe_and_self_inclusion_diagnostics():
    churn = universe_churn({"BTC", "ETH"}, {"ETH", "SOL"})
    assert churn.added == {"SOL"}
    assert churn.removed == {"BTC"}
    assert churn.jaccard_similarity == pytest.approx(1 / 3)

    predictions = {
        "jrai": np.array([0.0, 0.9, 0.2, 1.0]),
        "other": np.array([1.0, 0.4, 0.8, 0.0]),
    }
    weights = {"jrai": 3.0, "other": 1.0}
    full = weighted_meta_model(predictions, weights)
    leave_one_out = weighted_meta_model(
        predictions,
        weights,
        exclude_user="jrai",
    )
    diagnostic = self_inclusion_diagnostic("jrai", predictions, weights)

    assert not np.allclose(full, leave_one_out)
    assert diagnostic.mean_absolute_shift > 0
