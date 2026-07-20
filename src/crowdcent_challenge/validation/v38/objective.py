"""Percentile-calibrated CC Points surrogate for v3.8 selection."""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from math import ceil, cos, pi
from statistics import fmean

import numpy as np

from crowdcent_challenge.validation.atomic import (
    RAW_METRICS,
    UNIQUE_METRICS,
    AtomicPeriodScore,
)
from crowdcent_challenge.validation.eras import HYPERLIQUID_RANKING_ERAS, PointsObjective

POINTS_METRICS = RAW_METRICS + UNIQUE_METRICS


@dataclass(frozen=True)
class PeriodScore:
    period: date
    score: AtomicPeriodScore


def cosine_points(percentile: float) -> float:
    """Apply the production cosine curve to a percentile in ``[0, 100]``."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    return -cos(percentile * pi / 100.0) * 10.0


def slot_points_percentile(
    *,
    raw_percentile: float,
    unique_percentile: float,
    period: date,
    respect_live_era_cutover: bool,
) -> float:
    """Map calibrated slot percentiles to the book objective for ``period``."""

    if respect_live_era_cutover:
        objective = HYPERLIQUID_RANKING_ERAS.points_objective(period)
    else:
        objective = PointsObjective.RAW_UNIQUE_50_50
    if objective is PointsObjective.RAW_ONLY:
        return raw_percentile
    return 0.5 * raw_percentile + 0.5 * unique_percentile


@dataclass(frozen=True)
class MetricPercentileCalibrator:
    """Fixed empirical reference distributions for the eight scoring metrics."""

    sorted_reference: Mapping[str, np.ndarray]
    respect_live_era_cutover: bool = False

    def __post_init__(self) -> None:
        missing = set(POINTS_METRICS) - set(self.sorted_reference)
        if missing:
            raise ValueError(f"missing metric references: {sorted(missing)}")
        normalized = {}
        for metric in POINTS_METRICS:
            values = np.asarray(self.sorted_reference[metric], dtype=float)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"reference for {metric} must be a non-empty finite 1D array")
            normalized[metric] = np.sort(values)
        object.__setattr__(self, "sorted_reference", normalized)

    @classmethod
    def fit(
        cls,
        reference_scores: Iterable[AtomicPeriodScore],
        *,
        respect_live_era_cutover: bool = False,
    ) -> "MetricPercentileCalibrator":
        """Fit once on a stable inner-OOS reference pool."""

        values = {metric: [] for metric in POINTS_METRICS}
        count = 0
        for score in reference_scores:
            if score.unique is None:
                raise ValueError("percentile calibration requires uniqueness scores")
            count += 1
            for metric in RAW_METRICS:
                values[metric].append(score.raw[metric])
            for metric in UNIQUE_METRICS:
                values[metric].append(score.unique[metric])
        if not count:
            raise ValueError("at least one reference score is required")
        return cls(
            sorted_reference={
                metric: np.asarray(metric_values, dtype=float)
                for metric, metric_values in values.items()
            },
            respect_live_era_cutover=respect_live_era_cutover,
        )

    def percentile(self, metric: str, value: float) -> float:
        """Return an empirical midrank percentile against the fixed pool."""

        if metric not in self.sorted_reference:
            raise ValueError(f"unknown calibrated metric: {metric}")
        if not np.isfinite(value):
            raise ValueError("metric value must be finite")
        reference = self.sorted_reference[metric]
        left = np.searchsorted(reference, value, side="left")
        right = np.searchsorted(reference, value, side="right")
        return float(100.0 * (left + right) / (2.0 * len(reference)))

    def calibrate(
        self,
        *,
        candidate_id: Hashable,
        period: date,
        score: AtomicPeriodScore,
    ) -> "CalibratedSlotPercentile":
        if score.unique is None:
            raise ValueError("50/50 calibration requires uniqueness scores")
        raw_percentile = fmean(self.percentile(metric, score.raw[metric]) for metric in RAW_METRICS)
        unique_percentile = fmean(
            self.percentile(metric, score.unique[metric]) for metric in UNIQUE_METRICS
        )
        return CalibratedSlotPercentile(
            candidate_id=candidate_id,
            period=period,
            raw_percentile=raw_percentile,
            unique_percentile=unique_percentile,
            points_percentile=slot_points_percentile(
                raw_percentile=raw_percentile,
                unique_percentile=unique_percentile,
                period=period,
                respect_live_era_cutover=self.respect_live_era_cutover,
            ),
        )


@dataclass(frozen=True)
class CalibratedSlotPercentile:
    candidate_id: Hashable
    period: date
    raw_percentile: float
    unique_percentile: float
    points_percentile: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_percentile <= 100.0:
            raise ValueError("raw_percentile must be in [0, 100]")
        if not 0.0 <= self.unique_percentile <= 100.0:
            raise ValueError("unique_percentile must be in [0, 100]")
        if not 0.0 <= self.points_percentile <= 100.0:
            raise ValueError("points_percentile must be in [0, 100]")


@dataclass(frozen=True)
class BookPeriodUtility:
    period: date
    candidate_ids: tuple[Hashable, ...]
    points_percentile: float
    points: float


def book_period_utility(
    slot_percentiles: Iterable[CalibratedSlotPercentile],
) -> BookPeriodUtility:
    """Average selected slots first, then apply the nonlinear points curve."""

    slots = list(slot_percentiles)
    if not slots:
        raise ValueError("a book must contain at least one scored candidate")
    periods = {slot.period for slot in slots}
    if len(periods) != 1:
        raise ValueError("book slots must come from one period")
    candidate_ids = tuple(slot.candidate_id for slot in slots)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("a candidate cannot occupy multiple scored slots")
    percentile = fmean(slot.points_percentile for slot in slots)
    return BookPeriodUtility(
        period=slots[0].period,
        candidate_ids=candidate_ids,
        points_percentile=percentile,
        points=cosine_points(percentile),
    )


@dataclass(frozen=True)
class BootstrapMean:
    estimate: float
    lower: float
    upper: float
    block_length: int
    samples: int


def moving_block_bootstrap_weights(
    n_values: int,
    *,
    block_length: int = 30,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Return reusable period weights for moving-block bootstrap samples."""

    if n_values <= 0:
        raise ValueError("n_values must be positive")
    if block_length <= 0 or samples <= 0:
        raise ValueError("block_length and samples must be positive")

    # A circular block containing the full sample has the same mean at every
    # starting offset, which produces a falsely zero-width interval. Cap the
    # block at half the sample so every bootstrap draw contains at least two
    # independently selected blocks. The requested overlap-aware length is
    # retained whenever the sample is long enough.
    effective_block = min(block_length, max(1, n_values // 2))
    blocks_per_sample = ceil(n_values / effective_block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_values, size=(samples, blocks_per_sample))
    offsets = np.arange(effective_block)
    indices = (
        (starts[..., None] + offsets) % n_values
    ).reshape(samples, blocks_per_sample * effective_block)[:, :n_values]

    weights = np.zeros((samples, n_values), dtype=float)
    rows = np.repeat(np.arange(samples), n_values)
    np.add.at(weights, (rows, indices.reshape(-1)), 1.0 / n_values)
    return weights, effective_block


def moving_block_bootstrap_mean(
    values: Iterable[float],
    *,
    block_length: int = 30,
    samples: int = 2000,
    confidence: float = 0.80,
    seed: int = 42,
) -> BootstrapMean:
    """Estimate a mean interval while preserving overlapping-target dependence."""

    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite sequence")
    if block_length <= 0 or samples <= 0:
        raise ValueError("block_length and samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    weights, effective_block = moving_block_bootstrap_weights(
        len(array),
        block_length=block_length,
        samples=samples,
        seed=seed,
    )
    means = (
        np.full(samples, array[0], dtype=float)
        if np.all(array == array[0])
        else weights @ array
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapMean(
        estimate=float(array.mean()),
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        block_length=effective_block,
        samples=samples,
    )


def paired_book_delta(
    candidate: Iterable[BookPeriodUtility],
    incumbent: Iterable[BookPeriodUtility],
    *,
    block_length: int = 30,
    samples: int = 2000,
    confidence: float = 0.80,
    seed: int = 42,
) -> BootstrapMean:
    """Bootstrap matched date-level CC Points deltas."""

    candidate_by_period = {row.period: row.points for row in candidate}
    incumbent_by_period = {row.period: row.points for row in incumbent}
    if candidate_by_period.keys() != incumbent_by_period.keys():
        raise ValueError("candidate and incumbent books must have identical periods")
    deltas = [
        candidate_by_period[period] - incumbent_by_period[period]
        for period in sorted(candidate_by_period)
    ]
    return moving_block_bootstrap_mean(
        deltas,
        block_length=block_length,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
