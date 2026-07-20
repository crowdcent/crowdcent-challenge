"""Multiple-inner-fold chronology for unbiased v3.8 selection."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from crowdcent_challenge.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_folds,
)


@dataclass(frozen=True)
class V38NestedFoldConfig:
    outer_min_train_days: int = 730
    outer_validation_days: int = 120
    outer_step_days: int = 120
    outer_max_folds: int | None = None
    inner_min_train_days: int = 365
    inner_validation_days: int = 90
    inner_step_days: int = 90
    inner_folds: int = 3
    embargo_days: int = 31

    def __post_init__(self) -> None:
        durations = (
            self.outer_min_train_days,
            self.outer_validation_days,
            self.outer_step_days,
            self.inner_min_train_days,
            self.inner_validation_days,
            self.inner_step_days,
            self.inner_folds,
            self.embargo_days,
        )
        if min(durations) <= 0:
            raise ValueError("v3.8 fold durations and counts must be positive")


@dataclass(frozen=True)
class V38OuterFold:
    outer: WalkForwardFold
    inner: tuple[WalkForwardFold, ...]


def build_v38_nested_folds(
    dates: Iterable[date],
    config: V38NestedFoldConfig | None = None,
) -> list[V38OuterFold]:
    """Place several inner OOS folds wholly inside every outer train window."""

    config = config or V38NestedFoldConfig()
    unique_dates = sorted(set(dates))
    outer_folds = build_walk_forward_folds(
        unique_dates,
        WalkForwardConfig(
            min_train_days=config.outer_min_train_days,
            validation_days=config.outer_validation_days,
            embargo_days=config.embargo_days,
            step_days=config.outer_step_days,
            max_folds=config.outer_max_folds,
        ),
    )
    nested = []
    for outer in outer_folds:
        outer_train_dates = [
            period for period in unique_dates if outer.train_start <= period <= outer.train_end
        ]
        inner_folds = build_walk_forward_folds(
            outer_train_dates,
            WalkForwardConfig(
                min_train_days=config.inner_min_train_days,
                validation_days=config.inner_validation_days,
                embargo_days=config.embargo_days,
                step_days=config.inner_step_days,
                max_folds=config.inner_folds,
            ),
        )
        if len(inner_folds) < config.inner_folds:
            continue
        if any(fold.validation_end > outer.train_end for fold in inner_folds):
            raise AssertionError("inner validation escaped the outer training window")
        nested.append(V38OuterFold(outer=outer, inner=tuple(inner_folds)))
    return nested
