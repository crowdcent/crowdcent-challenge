from datetime import date, timedelta

import numpy as np
import pytest
from crowdcent_challenge.validation.atomic import (
    META_DIAGNOSTICS,
    RAW_METRICS,
    UNIQUE_METRICS,
    AtomicPeriodScore,
)
from crowdcent_challenge.validation.v38 import (
    BlendCandidate,
    BookPeriodUtility,
    BookSearchConfig,
    CalibratedSlotPercentile,
    MetaInputPanel,
    MetaScorePanel,
    MetricPercentileCalibrator,
    PredictionPanel,
    PrimitiveCandidate,
    ResidualCandidate,
    V38NestedFoldConfig,
    V38RunConfig,
    book_period_utility,
    build_candidate_panel,
    build_v38_nested_folds,
    candidate_library,
    cosine_points,
    moving_block_bootstrap_mean,
    paired_book_delta,
    require_same_keys,
    run_v38_smoke,
    select_scored_book,
)


def _prediction_panel(period=date(2026, 7, 2)):
    return PredictionPanel(
        periods=(period, period),
        ids=("BTC", "ETH"),
        values=np.array([[0.2, 0.4], [0.8, 0.6]]),
    )


def _atomic_score(value):
    return AtomicPeriodScore(
        raw={metric: value for metric in RAW_METRICS},
        unique={
            **{metric: value for metric in UNIQUE_METRICS},
            **{metric: 0.0 for metric in META_DIAGNOSTICS},
        },
    )


def test_v38_prediction_panels_preserve_keys_and_meta_roles():
    panel = _prediction_panel()
    require_same_keys(panel, _prediction_panel())

    MetaInputPanel(
        predictions=panel,
        release_dates=(date(2026, 7, 1), date(2026, 7, 1)),
    )
    MetaScorePanel(
        predictions=panel,
        release_dates=panel.periods,
    )

    with pytest.raises(ValueError, match="duplicate"):
        PredictionPanel(
            periods=(date(2026, 7, 2), date(2026, 7, 2)),
            ids=("BTC", "BTC"),
            values=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="strictly earlier"):
        MetaInputPanel(predictions=panel, release_dates=panel.periods)
    with pytest.raises(ValueError, match="exactly match"):
        MetaScorePanel(
            predictions=panel,
            release_dates=(date(2026, 7, 1), date(2026, 7, 1)),
        )


def test_metric_percentiles_are_fixed_and_blend_raw_unique_exactly():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    slot = calibrator.calibrate(
        candidate_id="candidate",
        period=date(2026, 7, 2),
        score=_atomic_score(0.5),
    )

    assert slot.raw_percentile == 50.0
    assert slot.unique_percentile == 50.0
    assert slot.points_percentile == 50.0
    assert book_period_utility([slot]).points == pytest.approx(0.0, abs=1e-12)


def test_book_average_precedes_cosine_curve():
    period = date(2026, 7, 2)
    weak = CalibratedSlotPercentile("weak", period, 20.0, 20.0, 20.0)
    strong = CalibratedSlotPercentile("strong", period, 80.0, 80.0, 80.0)
    utility = book_period_utility([weak, strong])

    assert utility.points_percentile == 50.0
    assert utility.points == pytest.approx(cosine_points(50.0))
    assert cosine_points(0.0) == -10.0
    assert cosine_points(100.0) == 10.0


def test_block_bootstrap_and_paired_delta_use_matched_periods():
    estimate = moving_block_bootstrap_mean(
        [1.0] * 60,
        block_length=30,
        samples=100,
    )
    assert estimate.estimate == estimate.lower == estimate.upper == 1.0
    assert estimate.block_length == 30

    start = date(2026, 1, 1)
    candidate = [
        BookPeriodUtility(
            period=start + timedelta(days=offset),
            candidate_ids=("new",),
            points_percentile=60.0,
            points=2.0,
        )
        for offset in range(60)
    ]
    incumbent = [
        BookPeriodUtility(
            period=start + timedelta(days=offset),
            candidate_ids=("old",),
            points_percentile=50.0,
            points=1.0,
        )
        for offset in range(60)
    ]
    delta = paired_book_delta(candidate, incumbent, samples=100)
    assert delta.estimate == delta.lower == delta.upper == 1.0


def test_block_bootstrap_caps_block_to_avoid_false_zero_width_interval():
    estimate = moving_block_bootstrap_mean(
        np.linspace(-1.0, 1.0, 25),
        block_length=30,
        samples=500,
        seed=7,
    )
    assert estimate.block_length == 12
    assert estimate.lower < estimate.estimate < estimate.upper


def test_candidates_use_keyed_components_and_prior_meta_only():
    panel = _prediction_panel()
    inverse = PredictionPanel(
        periods=panel.periods,
        ids=panel.ids,
        values=1.0 - panel.values,
    )
    components = {"xgb": panel, "transformer": inverse}
    blend = BlendCandidate(
        component_ids=("xgb", "transformer"),
        weights_10d=(0.75, 0.25),
        weights_30d=(0.25, 0.75),
    )
    blended = build_candidate_panel(blend, components)
    assert blended.keys == panel.keys

    residual = ResidualCandidate(
        base=PrimitiveCandidate("xgb"),
        strength_10d=0.5,
        strength_30d=0.5,
    )
    meta_input = MetaInputPanel(
        predictions=panel,
        release_dates=(date(2026, 7, 1), date(2026, 7, 1)),
    )
    assert (
        build_candidate_panel(
            residual,
            components,
            meta_input=meta_input,
        ).keys
        == panel.keys
    )

    scoring_meta = MetaScorePanel(predictions=panel, release_dates=panel.periods)
    with pytest.raises(TypeError, match="MetaInputPanel only"):
        build_candidate_panel(
            residual,
            components,
            meta_input=scoring_meta,  # type: ignore[arg-type]
        )


def test_book_search_rejects_a_weak_extra_slot():
    start = date(2026, 1, 1)
    periods = [start + timedelta(days=offset) for offset in range(60)]
    strong = [CalibratedSlotPercentile("strong", period, 70.0, 70.0, 70.0) for period in periods]
    weak = [CalibratedSlotPercentile("weak", period, 20.0, 20.0, 20.0) for period in periods]

    selection = select_scored_book(
        {"strong": strong, "weak": weak},
        incumbent_ids=("strong",),
        config=BookSearchConfig(bootstrap_samples=100),
    )

    assert selection.selected.candidate_ids == ("strong",)


def test_book_search_keeps_explicit_nominee_in_experimental_slot():
    start = date(2026, 1, 1)
    periods = [start + timedelta(days=offset) for offset in range(20)]
    rows = {
        candidate_id: [
            CalibratedSlotPercentile(
                candidate_id,
                period,
                percentile,
                percentile,
                percentile,
            )
            for period in periods
        ]
        for candidate_id, percentile in (
            ("strong", 70.0),
            ("weak", 20.0),
        )
    }

    selection = select_scored_book(
        rows,
        incumbent_ids=("strong",),
        experimental_ids=("weak",),
        config=BookSearchConfig(
            max_experimental_slots=1,
            bootstrap_samples=100,
        ),
    )

    assert selection.selected.candidate_ids == ("strong",)
    assert selection.experimental_ids == ("weak",)


def test_book_search_can_evaluate_a_fixed_scored_book():
    start = date(2026, 1, 1)
    periods = [start + timedelta(days=offset) for offset in range(20)]
    rows = {
        candidate_id: [
            CalibratedSlotPercentile(
                candidate_id,
                period,
                percentile,
                percentile,
                percentile,
            )
            for period in periods
        ]
        for candidate_id, percentile in (
            ("strong", 70.0),
            ("weak", 20.0),
        )
    }

    selection = select_scored_book(
        rows,
        incumbent_ids=("strong",),
        fixed_scored_ids=("weak",),
        config=BookSearchConfig(bootstrap_samples=100),
    )

    assert selection.selected.candidate_ids == ("weak",)
    assert selection.evaluations[0].candidate_ids == ("strong",)


def test_use50_counterfactual_applies_before_live_cutover():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    skewed = AtomicPeriodScore(
        raw={metric: 1.0 for metric in RAW_METRICS},
        unique={
            **{metric: 0.0 for metric in UNIQUE_METRICS},
            **{metric: 0.0 for metric in META_DIAGNOSTICS},
        },
    )
    pre_live = calibrator.calibrate(
        candidate_id="candidate",
        period=date(2026, 6, 1),
        score=skewed,
    )
    assert pre_live.points_percentile == pytest.approx(
        0.5 * pre_live.raw_percentile + 0.5 * pre_live.unique_percentile
    )

    live_era = MetricPercentileCalibrator.fit(
        [_atomic_score(0.0), _atomic_score(1.0)],
        respect_live_era_cutover=True,
    ).calibrate(
        candidate_id="candidate",
        period=date(2026, 6, 1),
        score=skewed,
    )
    assert live_era.points_percentile == pytest.approx(pre_live.raw_percentile)
    assert live_era.points_percentile != pytest.approx(pre_live.points_percentile)


def test_v38_uses_multiple_inner_folds_inside_each_outer_train_window():
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(1700)]
    folds = build_v38_nested_folds(
        dates,
        V38NestedFoldConfig(
            outer_min_train_days=800,
            outer_validation_days=90,
            outer_step_days=90,
            outer_max_folds=2,
            inner_min_train_days=365,
            inner_validation_days=60,
            inner_step_days=60,
            inner_folds=3,
            embargo_days=30,
        ),
    )

    assert len(folds) == 2
    for nested in folds:
        assert len(nested.inner) == 3
        assert nested.outer.embargo_days == 30
        assert all(inner.embargo_days == 30 for inner in nested.inner)
        assert all(inner.validation_end <= nested.outer.train_end for inner in nested.inner)


def test_candidate_library_never_includes_incumbent_slots():
    from crowdcent_challenge.validation.v38.incumbent import V36_SLOT_IDS

    for smoke in (True, False):
        specs = candidate_library(smoke=smoke)
        candidate_ids = {spec.candidate_id for spec in specs}
        assert not candidate_ids.intersection(V36_SLOT_IDS)


def test_smoke_run_excludes_incumbent_from_selected_book():
    result = run_v38_smoke(
        V38RunConfig(
            assets_per_period=24,
            n_features=12,
            seed=11,
        )
    )
    assert all(not str(candidate_id).startswith("v36:") for candidate_id in result.selected_candidate_ids)


def test_v38_smoke_runs_one_nested_fold_end_to_end():
    result = run_v38_smoke(
        V38RunConfig(
            assets_per_period=24,
            n_features=12,
            seed=11,
        )
    )

    assert result.outer_periods >= 5
    assert result.inner_periods >= 20
    assert result.selected_candidate_ids
    assert len(result.selected_candidate_ids) <= 5
    assert np.isfinite(result.outer_v38_points)
    assert np.isfinite(result.outer_v36_points)
    assert np.isfinite(result.outer_delta)
