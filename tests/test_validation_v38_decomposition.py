"""Tests for v3.8 first-principles score decomposition."""

from datetime import date, timedelta
from math import pi

import pytest
from crowdcent_challenge.validation.atomic import (
    META_DIAGNOSTICS,
    RAW_METRICS,
    UNIQUE_METRICS,
    AtomicPeriodScore,
)
from crowdcent_challenge.validation.v38 import (
    BookSearchConfig,
    MetricPercentileCalibrator,
    PeriodScore,
    V38RunConfig,
    cosine_points,
    cosine_slope,
    decompose_book,
    run_v38_smoke,
    select_scored_book,
)
from crowdcent_challenge.validation.v38.objective import CalibratedSlotPercentile


def _atomic_score(value: float) -> AtomicPeriodScore:
    return AtomicPeriodScore(
        raw={metric: value for metric in RAW_METRICS},
        unique={
            **{metric: value for metric in UNIQUE_METRICS},
            **{metric: 0.0 for metric in META_DIAGNOSTICS},
        },
    )


def _period_scores(candidate_id: str, start: date, count: int, value: float):
    return [
        PeriodScore(period=start + timedelta(days=offset), score=_atomic_score(value))
        for offset in range(count)
    ]


def test_metric_contributions_sum_to_book_percentile():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    start = date(2026, 1, 1)
    atomic = {
        "strong": _period_scores("strong", start, 30, 0.8),
        "weak": _period_scores("weak", start, 30, 0.2),
    }
    decomposition = decompose_book(
        calibrator,
        atomic,
        scored_ids=("strong", "weak"),
        scope="outer",
    )
    total = sum(row.percentile_contribution for row in decomposition.metric_contributions)
    assert len(decomposition.metric_contributions) == 8
    assert total == pytest.approx(decomposition.book_points_percentile, abs=1e-9)


def test_raw_unique_split_sums_to_book_percentile():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    start = date(2026, 1, 1)
    decomposition = decompose_book(
        calibrator,
        {"solo": _period_scores("solo", start, 20, 0.6)},
        scored_ids=("solo",),
        scope="outer",
    )
    split = decomposition.raw_unique_split
    assert split.raw_contribution + split.unique_contribution == pytest.approx(
        decomposition.book_points_percentile,
        abs=1e-9,
    )


def test_loo_delta_flags_dragging_scored_slot():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    start = date(2026, 1, 1)
    decomposition = decompose_book(
        calibrator,
        {
            "strong": _period_scores("strong", start, 30, 1.0),
            "weak": _period_scores("weak", start, 30, 0.0),
        },
        scored_ids=("strong", "weak"),
        scope="outer",
    )
    by_id = {row.candidate_id: row for row in decomposition.slot_contributions}
    assert by_id["weak"].loo_points_delta is not None
    assert by_id["weak"].loo_points_delta < 0
    assert by_id["strong"].loo_points_delta is not None
    assert by_id["strong"].loo_points_delta > 0


def test_experimental_excluded_from_points_but_scored_for_promotion():
    calibrator = MetricPercentileCalibrator.fit([_atomic_score(0.0), _atomic_score(1.0)])
    start = date(2026, 1, 1)
    atomic = {
        "scored": _period_scores("scored", start, 25, 0.7),
        "shadow": _period_scores("shadow", start, 25, 0.75),
    }
    scored_only = decompose_book(
        calibrator,
        atomic,
        scored_ids=("scored",),
        scope="outer",
    )
    with_shadow = decompose_book(
        calibrator,
        atomic,
        scored_ids=("scored",),
        experimental_ids=("shadow",),
        scope="outer",
    )
    assert with_shadow.book_points == pytest.approx(scored_only.book_points)
    experimental = next(
        row for row in with_shadow.slot_contributions if row.role == "experimental"
    )
    assert experimental.share_of_book is None
    assert experimental.promotion is not None


def test_book_search_respects_min_scored_and_experimental_slots():
    start = date(2026, 1, 1)
    periods = [start + timedelta(days=offset) for offset in range(60)]
    rows = {
        "strong": [
            CalibratedSlotPercentile("strong", period, 70.0, 70.0, 70.0) for period in periods
        ],
        "mid": [
            CalibratedSlotPercentile("mid", period, 55.0, 55.0, 55.0) for period in periods
        ],
        "weak": [
            CalibratedSlotPercentile("weak", period, 20.0, 20.0, 20.0) for period in periods
        ],
    }
    selection = select_scored_book(
        rows,
        incumbent_ids=("strong",),
        config=BookSearchConfig(
            min_scored_slots=1,
            max_experimental_slots=1,
            bootstrap_samples=100,
        ),
    )
    assert len(selection.selected.candidate_ids) >= 1
    assert len(selection.experimental_ids) <= 1
    assert set(selection.experimental_ids).isdisjoint(selection.selected.candidate_ids)


def test_cosine_slope_matches_finite_difference():
    assert cosine_slope(50.0) == pytest.approx(pi / 10.0)
    delta = 1e-5
    numeric = (cosine_points(50.0 + delta) - cosine_points(50.0 - delta)) / (2 * delta)
    assert cosine_slope(50.0) == pytest.approx(numeric, abs=1e-4)


def test_smoke_run_includes_decomposition():
    result = run_v38_smoke(
        V38RunConfig(
            assets_per_period=24,
            n_features=12,
            seed=11,
        )
    )
    assert result.outer_decomposition.scope == "outer"
    assert result.inner_decomposition.scope == "inner"
    assert result.outer_decomposition.metric_contributions
