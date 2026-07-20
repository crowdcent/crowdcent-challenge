"""Time-aware rules for Hyperliquid Ranking research.

The challenge has three distinct clocks:

* training labels are available from 2020;
* tournament submissions and point-in-time meta-model snapshots start in 2025;
* CC Points changed from raw-only to 50/50 raw and unique in 2026.

Keeping these dates in one object prevents validation code from silently using
the current scoring regime or today's meta-model in an earlier era.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class PointsObjective(str, Enum):
    """The performance objective attached to an inference period."""

    RAW_ONLY = "raw_only"
    RAW_UNIQUE_50_50 = "raw_unique_50_50"


@dataclass(frozen=True)
class ChallengeEras:
    """Canonical dates that constrain Hyperliquid Ranking validation."""

    training_start: date
    tournament_start: date
    meta_model_start: date
    use50_points_start: date
    performance_award_lag_days: int = 31

    def has_tournament_context(self, period: date) -> bool:
        return period >= self.tournament_start

    def has_meta_model(self, period: date) -> bool:
        return period >= self.meta_model_start

    def points_objective(self, period: date) -> PointsObjective:
        if period >= self.use50_points_start:
            return PointsObjective.RAW_UNIQUE_50_50
        return PointsObjective.RAW_ONLY

    def performance_award_date(self, period: date) -> date:
        """Date when both horizons are final and performance points are written."""

        return period + timedelta(days=self.performance_award_lag_days)

    def is_fully_resolved(self, period: date, as_of: date) -> bool:
        return as_of >= self.performance_award_date(period)

    def validate_training_period(self, period: date) -> None:
        if period < self.training_start:
            raise ValueError(
                f"{period} predates the supported training era ({self.training_start})"
            )

    def validate_meta_period(self, period: date) -> None:
        if not self.has_meta_model(period):
            raise ValueError(
                f"{period} predates point-in-time meta-model history ({self.meta_model_start})"
            )


HYPERLIQUID_RANKING_ERAS = ChallengeEras(
    training_start=date(2020, 1, 1),
    tournament_start=date(2025, 6, 5),
    meta_model_start=date(2025, 6, 5),
    use50_points_start=date(2026, 7, 1),
)
