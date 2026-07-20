"""Evidence gates for experimental-slot promotion."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from statistics import fmean


@dataclass(frozen=True)
class ShadowPercentile:
    """One live shadow observation for an experimental slot."""

    period: date
    raw_percentile: float
    unique_percentile: float
    resolved: bool

    def __post_init__(self) -> None:
        for name in ("raw_percentile", "unique_percentile"):
            value = getattr(self, name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")

    @property
    def blended_percentile(self) -> float:
        return 0.5 * self.raw_percentile + 0.5 * self.unique_percentile


@dataclass(frozen=True)
class PromotionPolicy:
    """Precommitted requirements for changing the scored book."""

    min_resolved_periods: int = 21
    min_mean_margin: float = 0.0
    max_underperformance_frequency: float = 0.45

    def __post_init__(self) -> None:
        if self.min_resolved_periods <= 0:
            raise ValueError("min_resolved_periods must be positive")
        if not 0.0 <= self.max_underperformance_frequency <= 1.0:
            raise ValueError("max_underperformance_frequency must be in [0, 1]")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    matched_periods: int
    candidate_mean: float | None
    incumbent_mean: float | None
    underperformance_frequency: float | None
    reasons: tuple[str, ...]


def evaluate_shadow_promotion(
    observations: list[ShadowPercentile],
    incumbent_book_percentiles: Mapping[date, float],
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Compare a shadow slot with the concurrent scored-book percentile.

    Only fully resolved periods with a same-period incumbent observation count.
    This avoids mixing provisional 10d/30d values or comparing different market
    regimes.
    """

    policy = policy or PromotionPolicy()
    matched = [
        (observation.blended_percentile, incumbent_book_percentiles[observation.period])
        for observation in observations
        if observation.resolved and observation.period in incumbent_book_percentiles
    ]
    reasons: list[str] = []
    if len(matched) < policy.min_resolved_periods:
        reasons.append(
            f"need {policy.min_resolved_periods} resolved matched periods; found {len(matched)}"
        )
        return PromotionDecision(
            promote=False,
            matched_periods=len(matched),
            candidate_mean=None,
            incumbent_mean=None,
            underperformance_frequency=None,
            reasons=tuple(reasons),
        )

    candidate_mean = fmean(candidate for candidate, _ in matched)
    incumbent_mean = fmean(incumbent for _, incumbent in matched)
    underperformance_frequency = sum(
        candidate < incumbent for candidate, incumbent in matched
    ) / len(matched)
    if candidate_mean < incumbent_mean + policy.min_mean_margin:
        reasons.append(
            f"mean margin {candidate_mean - incumbent_mean:.2f} is below "
            f"{policy.min_mean_margin:.2f}"
        )
    if underperformance_frequency > policy.max_underperformance_frequency:
        reasons.append(
            f"underperformance frequency {underperformance_frequency:.1%} exceeds "
            f"{policy.max_underperformance_frequency:.1%}"
        )

    return PromotionDecision(
        promote=not reasons,
        matched_periods=len(matched),
        candidate_mean=candidate_mean,
        incumbent_mean=incumbent_mean,
        underperformance_frequency=underperformance_frequency,
        reasons=tuple(reasons),
    )
