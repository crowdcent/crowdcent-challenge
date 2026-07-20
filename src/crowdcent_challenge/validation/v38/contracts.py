"""Point-in-time data contracts for the first-principles v3.8 pipeline."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from crowdcent_challenge.validation.training.features import SequenceFeatureSchema


@dataclass(frozen=True)
class PreparedDataset:
    """Row-aligned training panel for walk-forward evaluation."""

    periods: tuple[date, ...]
    ids: tuple[str, ...]
    targets: np.ndarray
    features: np.ndarray
    sequence_features: np.ndarray | None = None
    sequence_schema: SequenceFeatureSchema | None = None

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets, dtype=float)
        features = np.asarray(self.features, dtype=float)
        row_count = len(self.periods)
        if len(self.ids) != row_count or targets.shape != (row_count, 2):
            raise ValueError("periods, ids, and targets must align as (rows, 2)")
        if features.shape[0] != row_count or features.ndim != 2:
            raise ValueError("features must have shape (rows, n_features)")
        if len(set(zip(self.periods, self.ids, strict=True))) != row_count:
            raise ValueError("prepared dataset contains duplicate (period, id) keys")
        if self.sequence_features is not None:
            sequence = np.asarray(self.sequence_features, dtype=np.float32)
            if sequence.ndim != 2 or sequence.shape[0] != row_count:
                raise ValueError("sequence_features must have shape (rows, seq_len * n_features)")
            if self.sequence_schema is not None:
                expected = (
                    len(self.sequence_schema.lag_windows)
                    * self.sequence_schema.n_features_per_timestep
                )
                if sequence.shape[1] != expected:
                    raise ValueError("sequence_features width does not match schema")
            object.__setattr__(self, "sequence_features", sequence)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "features", features)

    @property
    def unique_periods(self) -> tuple[date, ...]:
        return tuple(sorted(set(self.periods)))

    def mask_between(self, start: date, end: date) -> np.ndarray:
        return np.asarray([start <= period <= end for period in self.periods])

    def take(self, mask: np.ndarray) -> PreparedDataset:
        positions = np.flatnonzero(mask)
        return PreparedDataset(
            periods=tuple(self.periods[index] for index in positions),
            ids=tuple(self.ids[index] for index in positions),
            targets=self.targets[positions],
            features=self.features[positions],
            sequence_features=(
                self.sequence_features[positions] if self.sequence_features is not None else None
            ),
            sequence_schema=self.sequence_schema,
        )


@dataclass(frozen=True)
class PredictionPanel:
    """Two-horizon predictions keyed by unique ``(period, id)`` rows."""

    periods: tuple[date, ...]
    ids: tuple[Hashable, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        row_count = len(self.periods)
        if len(self.ids) != row_count or values.shape != (row_count, 2):
            raise ValueError("periods, ids, and values must align with values shaped (rows, 2)")
        if len(set(zip(self.periods, self.ids, strict=True))) != row_count:
            raise ValueError("prediction panel contains duplicate (period, id) keys")
        if not np.isfinite(values).all():
            raise ValueError("prediction values must be finite")
        object.__setattr__(self, "values", values)

    @property
    def keys(self) -> tuple[tuple[date, Hashable], ...]:
        return tuple(zip(self.periods, self.ids, strict=True))

    def take(self, indices: np.ndarray) -> PredictionPanel:
        positions = np.asarray(indices, dtype=int)
        return PredictionPanel(
            periods=tuple(self.periods[index] for index in positions),
            ids=tuple(self.ids[index] for index in positions),
            values=self.values[positions],
        )


@dataclass(frozen=True)
class MetaInputPanel:
    """Prior meta releases that were observable before prediction time."""

    predictions: PredictionPanel
    release_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if len(self.release_dates) != len(self.predictions.periods):
            raise ValueError("meta-input release dates must align with predictions")
        invalid = [
            (release, period)
            for release, period in zip(
                self.release_dates,
                self.predictions.periods,
                strict=True,
            )
            if release >= period
        ]
        if invalid:
            raise ValueError("meta-input releases must be strictly earlier than prediction periods")


@dataclass(frozen=True)
class MetaScorePanel:
    """Same-period meta releases used only to measure uniqueness."""

    predictions: PredictionPanel
    release_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if len(self.release_dates) != len(self.predictions.periods):
            raise ValueError("scoring-meta release dates must align with predictions")
        if any(
            release != period
            for release, period in zip(
                self.release_dates,
                self.predictions.periods,
                strict=True,
            )
        ):
            raise ValueError("scoring-meta releases must exactly match prediction periods")


def require_same_keys(*panels: PredictionPanel) -> None:
    """Fail before arithmetic if independently produced panels are misaligned."""

    if not panels:
        raise ValueError("at least one prediction panel is required")
    expected = panels[0].keys
    if any(panel.keys != expected for panel in panels[1:]):
        raise ValueError("prediction panels do not have identical ordered keys")
