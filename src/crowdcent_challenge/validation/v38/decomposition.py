"""First-principles atoms-to-points decomposition for a v3.8 book."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import pi, sin
from statistics import fmean

from crowdcent_challenge.validation.atomic import RAW_METRICS, UNIQUE_METRICS
from crowdcent_challenge.validation.promotion import (
    PromotionDecision,
    PromotionPolicy,
    ShadowPercentile,
    evaluate_shadow_promotion,
)

from .objective import (
    POINTS_METRICS,
    CalibratedSlotPercentile,
    MetricPercentileCalibrator,
    PeriodScore,
    book_period_utility,
)

METRIC_WEIGHT = 0.125


def cosine_slope(percentile: float) -> float:
    """Marginal performance points per percentile-point at ``percentile``."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    return (pi / 10.0) * sin(percentile * pi / 100.0)


@dataclass(frozen=True)
class MetricContribution:
    metric: str
    side: str
    mean_percentile: float
    lift_vs_neutral: float
    weight: float
    percentile_contribution: float


@dataclass(frozen=True)
class RawUniqueSplit:
    raw_percentile: float
    unique_percentile: float
    raw_contribution: float
    unique_contribution: float


@dataclass(frozen=True)
class SlotContribution:
    candidate_id: Hashable
    role: str
    mean_points_percentile: float
    raw_percentile: float
    unique_percentile: float
    share_of_book: float | None
    loo_book_points: float | None
    loo_points_delta: float | None
    promotion: PromotionDecision | None


@dataclass(frozen=True)
class BookScoreDecomposition:
    scope: str
    periods: int
    scored_candidate_ids: tuple[Hashable, ...]
    experimental_candidate_ids: tuple[Hashable, ...]
    book_points_percentile: float
    book_points: float
    cosine_local_slope: float
    metric_contributions: tuple[MetricContribution, ...]
    raw_unique_split: RawUniqueSplit
    slot_contributions: tuple[SlotContribution, ...]
    base_streak_points: float = 0.5


def decompose_book(
    calibrator: MetricPercentileCalibrator,
    atomic_by_candidate: Mapping[Hashable, Sequence[PeriodScore]],
    *,
    scored_ids: tuple[Hashable, ...],
    experimental_ids: tuple[Hashable, ...] = (),
    scope: str = "outer",
    promotion_policy: PromotionPolicy | None = None,
) -> BookScoreDecomposition:
    if not scored_ids:
        raise ValueError("at least one scored candidate is required")
    missing = set(scored_ids) - set(atomic_by_candidate)
    if missing:
        raise ValueError(f"missing atomic scores for scored candidates: {sorted(missing)}")

    calibrated_rows: dict[Hashable, dict[date, tuple[CalibratedSlotPercentile, dict[str, float]]]] = {}
    for candidate_id, period_scores in atomic_by_candidate.items():
        by_period: dict[date, tuple[CalibratedSlotPercentile, dict[str, float]]] = {}
        for period_score in period_scores:
            score = period_score.score
            if score.unique is None:
                continue
            metric_percentiles = {}
            for metric in RAW_METRICS:
                metric_percentiles[metric] = calibrator.percentile(metric, score.raw[metric])
            for metric in UNIQUE_METRICS:
                metric_percentiles[metric] = calibrator.percentile(metric, score.unique[metric])
            slot = calibrator.calibrate(
                candidate_id=candidate_id,
                period=period_score.period,
                score=score,
            )
            by_period[period_score.period] = (slot, metric_percentiles)
        calibrated_rows[candidate_id] = by_period

    common_periods = None
    for candidate_id in scored_ids:
        periods = set(calibrated_rows[candidate_id])
        common_periods = periods if common_periods is None else common_periods & periods
    common_periods = tuple(sorted(common_periods or ()))
    if not common_periods:
        raise ValueError("scored candidates have no common calibrated periods")

    per_period_utilities = [
        book_period_utility(
            [calibrated_rows[candidate_id][period][0] for candidate_id in scored_ids]
        )
        for period in common_periods
    ]
    book_points = fmean(row.points for row in per_period_utilities)
    book_points_percentile = fmean(row.points_percentile for row in per_period_utilities)
    incumbent_by_period = {row.period: row.points_percentile for row in per_period_utilities}

    metric_contributions = []
    for metric in POINTS_METRICS:
        side = "raw" if metric in RAW_METRICS else "unique"
        mean_percentile = fmean(
            calibrated_rows[candidate_id][period][1][metric]
            for candidate_id in scored_ids
            for period in common_periods
        )
        metric_contributions.append(
            MetricContribution(
                metric=metric,
                side=side,
                mean_percentile=mean_percentile,
                lift_vs_neutral=mean_percentile - 50.0,
                weight=METRIC_WEIGHT,
                percentile_contribution=METRIC_WEIGHT * mean_percentile,
            )
        )
    contribution_sum = sum(row.percentile_contribution for row in metric_contributions)
    if abs(contribution_sum - book_points_percentile) > 1e-6:
        raise AssertionError(
            f"metric contributions {contribution_sum:.6f} != book percentile "
            f"{book_points_percentile:.6f}"
        )

    raw_percentile = fmean(
        calibrated_rows[candidate_id][period][0].raw_percentile
        for candidate_id in scored_ids
        for period in common_periods
    )
    unique_percentile = fmean(
        calibrated_rows[candidate_id][period][0].unique_percentile
        for candidate_id in scored_ids
        for period in common_periods
    )
    raw_unique_split = RawUniqueSplit(
        raw_percentile=raw_percentile,
        unique_percentile=unique_percentile,
        raw_contribution=0.5 * raw_percentile,
        unique_contribution=0.5 * unique_percentile,
    )

    slot_contributions: list[SlotContribution] = []
    for candidate_id in scored_ids:
        slots = [calibrated_rows[candidate_id][period][0] for period in common_periods]
        loo_book_points = None
        loo_points_delta = None
        if len(scored_ids) > 1:
            remaining = tuple(other for other in scored_ids if other != candidate_id)
            loo_utilities = [
                book_period_utility([calibrated_rows[other][period][0] for other in remaining])
                for period in common_periods
            ]
            loo_book_points = fmean(row.points for row in loo_utilities)
            loo_points_delta = book_points - loo_book_points
        slot_contributions.append(
            SlotContribution(
                candidate_id=candidate_id,
                role="scored",
                mean_points_percentile=fmean(slot.points_percentile for slot in slots),
                raw_percentile=fmean(slot.raw_percentile for slot in slots),
                unique_percentile=fmean(slot.unique_percentile for slot in slots),
                share_of_book=1.0 / len(scored_ids),
                loo_book_points=loo_book_points,
                loo_points_delta=loo_points_delta,
                promotion=None,
            )
        )

    policy = promotion_policy or PromotionPolicy()
    for candidate_id in experimental_ids:
        if candidate_id not in calibrated_rows:
            continue
        by_period = calibrated_rows[candidate_id]
        shadows = [
            ShadowPercentile(
                period=period,
                raw_percentile=by_period[period][0].raw_percentile,
                unique_percentile=by_period[period][0].unique_percentile,
                resolved=True,
            )
            for period in common_periods
            if period in by_period
        ]
        slots = [by_period[period][0] for period in common_periods if period in by_period]
        slot_contributions.append(
            SlotContribution(
                candidate_id=candidate_id,
                role="experimental",
                mean_points_percentile=fmean(slot.points_percentile for slot in slots),
                raw_percentile=fmean(slot.raw_percentile for slot in slots),
                unique_percentile=fmean(slot.unique_percentile for slot in slots),
                share_of_book=None,
                loo_book_points=None,
                loo_points_delta=None,
                promotion=evaluate_shadow_promotion(
                    shadows,
                    incumbent_by_period,
                    policy,
                ),
            )
        )

    return BookScoreDecomposition(
        scope=scope,
        periods=len(common_periods),
        scored_candidate_ids=scored_ids,
        experimental_candidate_ids=experimental_ids,
        book_points_percentile=book_points_percentile,
        book_points=book_points,
        cosine_local_slope=cosine_slope(book_points_percentile),
        metric_contributions=tuple(metric_contributions),
        raw_unique_split=raw_unique_split,
        slot_contributions=tuple(slot_contributions),
    )
