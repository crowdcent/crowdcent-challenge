"""Leakage-controlled date folds for ranking-model validation."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class WalkForwardConfig:
    """Calendar-based walk-forward settings."""

    min_train_days: int = 730
    validation_days: int = 120
    embargo_days: int = 30
    step_days: int | None = None
    rolling_train_days: int | None = None
    max_folds: int | None = None

    def __post_init__(self) -> None:
        for name in ("min_train_days", "validation_days", "embargo_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_days is not None and self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.rolling_train_days is not None:
            if self.rolling_train_days < self.min_train_days:
                raise ValueError("rolling_train_days cannot be shorter than min_train_days")
        if self.max_folds is not None and self.max_folds <= 0:
            raise ValueError("max_folds must be positive")


@dataclass(frozen=True)
class WalkForwardFold:
    """Inclusive train and validation date boundaries."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date

    @property
    def embargo_days(self) -> int:
        return (self.validation_start - self.train_end).days - 1

    def train_mask(self, dates: Iterable[date]) -> list[bool]:
        return [self.train_start <= value <= self.train_end for value in dates]

    def validation_mask(self, dates: Iterable[date]) -> list[bool]:
        return [self.validation_start <= value <= self.validation_end for value in dates]


def build_walk_forward_folds(
    dates: Iterable[date],
    config: WalkForwardConfig | None = None,
) -> list[WalkForwardFold]:
    """Build chronological, non-overlapping validation folds.

    The final training date is at least ``embargo_days`` full calendar days
    before validation begins. Validation advances by ``step_days`` or one full
    validation window. When ``max_folds`` is set, the most recent folds are
    retained.
    """

    config = config or WalkForwardConfig()
    unique_dates = sorted(set(dates))
    if not unique_dates:
        return []

    first_date = unique_dates[0]
    last_date = unique_dates[-1]
    validation_start = (
        first_date + timedelta(days=config.min_train_days) + timedelta(days=config.embargo_days + 1)
    )
    step = config.step_days or config.validation_days
    folds: list[WalkForwardFold] = []

    while validation_start <= last_date:
        validation_end = min(
            validation_start + timedelta(days=config.validation_days - 1),
            last_date,
        )
        train_end = validation_start - timedelta(days=config.embargo_days + 1)
        train_start = first_date
        if config.rolling_train_days is not None:
            train_start = max(
                first_date,
                train_end - timedelta(days=config.rolling_train_days - 1),
            )

        train_count = sum(train_start <= value <= train_end for value in unique_dates)
        validation_count = sum(
            validation_start <= value <= validation_end for value in unique_dates
        )
        if train_count and validation_count:
            folds.append(
                WalkForwardFold(
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                )
            )
        validation_start += timedelta(days=step)

    if config.max_folds is not None:
        return folds[-config.max_folds :]
    return folds
