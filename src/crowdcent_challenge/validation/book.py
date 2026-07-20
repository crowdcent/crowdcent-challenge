"""Resolved live-percentile analysis for scored-slot selection."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from math import cos, pi
from statistics import fmean, pstdev


@dataclass(frozen=True)
class LiveSlotPercentile:
    """Raw and unique percentiles observed for one slot and period."""

    period: date
    slot: int
    raw_percentile: float
    unique_percentile: float
    resolved: bool

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("slot must be positive")
        for name in ("raw_percentile", "unique_percentile"):
            value = getattr(self, name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")

    @property
    def blended_percentile(self) -> float:
        return 0.5 * self.raw_percentile + 0.5 * self.unique_percentile


def performance_adjustment(points_percentile: float) -> float:
    """Apply the exact CC Points cosine performance curve."""

    if not 0.0 <= points_percentile <= 100.0:
        raise ValueError("points_percentile must be in [0, 100]")
    return -cos(points_percentile * pi / 100.0) * 10.0


@dataclass(frozen=True)
class ScoredSubsetResult:
    slots: tuple[int, ...]
    periods: int
    mean_percentile: float
    mean_performance_adjustment: float
    performance_volatility: float
    negative_adjustment_frequency: float
    worst_adjustment: float


def evaluate_scored_subsets(
    observations: Iterable[LiveSlotPercentile],
    *,
    candidate_slots: set[int] | None = None,
    min_resolved_periods: int = 21,
) -> list[ScoredSubsetResult]:
    """Rank every non-empty slot subset on common, fully resolved periods.

    All subsets use the same complete-period panel, avoiding an apples-to-oranges
    comparison caused by missing slots. Results are descriptive rather than an
    automatic promotion rule because changing experimental flags also changes
    the live percentile pool and meta-model.
    """

    if min_resolved_periods <= 0:
        raise ValueError("min_resolved_periods must be positive")
    rows = [observation for observation in observations if observation.resolved]
    available_slots = {observation.slot for observation in rows}
    selected_slots = available_slots if candidate_slots is None else set(candidate_slots)
    if not selected_slots:
        raise ValueError("at least one candidate slot is required")
    missing = selected_slots - available_slots
    if missing:
        raise ValueError(f"no resolved observations for slots: {sorted(missing)}")

    by_period: dict[date, dict[int, float]] = defaultdict(dict)
    for observation in rows:
        if observation.slot in selected_slots:
            by_period[observation.period][observation.slot] = observation.blended_percentile
    complete_periods = {
        period: values for period, values in by_period.items() if selected_slots.issubset(values)
    }
    if len(complete_periods) < min_resolved_periods:
        raise ValueError(
            f"need {min_resolved_periods} common resolved periods; found {len(complete_periods)}"
        )

    results: list[ScoredSubsetResult] = []
    ordered_slots = sorted(selected_slots)
    for size in range(1, len(ordered_slots) + 1):
        for subset in combinations(ordered_slots, size):
            daily_percentiles = [
                fmean(values[slot] for slot in subset) for values in complete_periods.values()
            ]
            adjustments = [performance_adjustment(percentile) for percentile in daily_percentiles]
            results.append(
                ScoredSubsetResult(
                    slots=subset,
                    periods=len(complete_periods),
                    mean_percentile=fmean(daily_percentiles),
                    mean_performance_adjustment=fmean(adjustments),
                    performance_volatility=pstdev(adjustments),
                    negative_adjustment_frequency=(
                        sum(adjustment < 0 for adjustment in adjustments) / len(adjustments)
                    ),
                    worst_adjustment=min(adjustments),
                )
            )
    return sorted(
        results,
        key=lambda result: (
            result.mean_performance_adjustment,
            -result.performance_volatility,
        ),
        reverse=True,
    )
