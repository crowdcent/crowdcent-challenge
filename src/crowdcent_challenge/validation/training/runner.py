"""Nested walk-forward boundaries for full-fidelity model calibration."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from crowdcent_challenge.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_folds,
)


@dataclass(frozen=True)
class NestedCalibrationConfig:
    """Inner calibration tail placed wholly inside each outer train fold."""

    calibration_days: int = 120
    embargo_days: int = 30
    min_inner_train_days: int = 365

    def __post_init__(self) -> None:
        if (
            min(
                self.calibration_days,
                self.embargo_days,
                self.min_inner_train_days,
            )
            <= 0
        ):
            raise ValueError("nested calibration durations must be positive")


@dataclass(frozen=True)
class NestedWalkForwardFold:
    """Outer OOS fold plus an inner, embargoed tuning split."""

    outer: WalkForwardFold
    inner_train_start: date
    inner_train_end: date
    calibration_start: date
    calibration_end: date

    @property
    def inner_embargo_days(self) -> int:
        return (self.calibration_start - self.inner_train_end).days - 1


def build_nested_walk_forward_folds(
    dates: Iterable[date],
    *,
    outer_config: WalkForwardConfig | None = None,
    calibration_config: NestedCalibrationConfig | None = None,
) -> list[NestedWalkForwardFold]:
    """Build folds where tuning never touches the outer validation period."""

    unique_dates = sorted(set(dates))
    if not unique_dates:
        return []
    outer_config = outer_config or WalkForwardConfig()
    calibration_config = calibration_config or NestedCalibrationConfig()
    outer_folds = build_walk_forward_folds(unique_dates, outer_config)
    nested = []
    for outer in outer_folds:
        calibration_end = outer.train_end
        calibration_start = calibration_end - timedelta(
            days=calibration_config.calibration_days - 1
        )
        inner_train_end = calibration_start - timedelta(days=calibration_config.embargo_days + 1)
        inner_train_start = outer.train_start
        if (inner_train_end - inner_train_start).days + 1 < calibration_config.min_inner_train_days:
            continue

        inner_count = sum(inner_train_start <= period <= inner_train_end for period in unique_dates)
        calibration_count = sum(
            calibration_start <= period <= calibration_end for period in unique_dates
        )
        if not inner_count or not calibration_count:
            continue
        nested.append(
            NestedWalkForwardFold(
                outer=outer,
                inner_train_start=inner_train_start,
                inner_train_end=inner_train_end,
                calibration_start=calibration_start,
                calibration_end=calibration_end,
            )
        )
    return nested
