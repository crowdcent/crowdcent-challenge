"""Atomic Hyperliquid metrics and offline objective proxies.

Live CC Points use participant percentiles, which are endogenous to the daily
submission pool and cannot be replayed exactly for a counterfactual model.
This module deliberately stays in score space: four raw metrics, four unique
metrics, and their equal-weight proxy. Live shadow percentiles remain the
promotion authority.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean

import numpy as np
from scipy.stats import rankdata

from crowdcent_challenge.scoring import (
    evaluate_hyperliquid_submission,
    evaluate_hyperliquid_uniqueness,
)

from .eras import PointsObjective

RAW_METRICS = (
    "spearman_10d",
    "spearman_30d",
    "ndcg@40_10d",
    "ndcg@40_30d",
)
UNIQUE_METRICS = (
    "unique_spearman_10d",
    "unique_spearman_30d",
    "unique_ndcg@40_10d",
    "unique_ndcg@40_30d",
)
META_DIAGNOSTICS = ("corr_to_meta_10d", "corr_to_meta_30d")


class InsufficientAssetsError(ValueError):
    """Raised when a cross-section cannot support challenge-style scoring."""


def uniform_rank(values: np.ndarray) -> np.ndarray:
    """Return average-tie uniform ranks in ``[0, 1]``."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(array) == 0:
        return array
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite numbers")
    if len(array) == 1:
        return np.array([0.5])
    return (rankdata(array, method="average") - 1.0) / (len(array) - 1.0)


@dataclass(frozen=True)
class AtomicPeriodScore:
    """Exact atomic scores for one inference-period cross-section."""

    raw: Mapping[str, float]
    unique: Mapping[str, float] | None = None

    @property
    def raw_composite(self) -> float:
        return fmean(self.raw[name] for name in RAW_METRICS)

    @property
    def unique_composite(self) -> float | None:
        if self.unique is None:
            return None
        return fmean(self.unique[name] for name in UNIQUE_METRICS)

    @property
    def use50_proxy(self) -> float | None:
        unique = self.unique_composite
        if unique is None:
            return None
        return 0.5 * self.raw_composite + 0.5 * unique

    def objective_proxy(self, objective: PointsObjective) -> float:
        if objective is PointsObjective.RAW_ONLY:
            return self.raw_composite
        use50 = self.use50_proxy
        if use50 is None:
            raise ValueError("50/50 objective requires a point-in-time meta-model")
        return use50

    @property
    def corr_to_meta(self) -> tuple[float, float] | None:
        if self.unique is None:
            return None
        return tuple(self.unique[name] for name in META_DIAGNOSTICS)


def score_atomic_period(
    *,
    y_true_10d: np.ndarray,
    y_pred_10d: np.ndarray,
    y_true_30d: np.ndarray,
    y_pred_30d: np.ndarray,
    meta_pred_10d: np.ndarray | None = None,
    meta_pred_30d: np.ndarray | None = None,
    min_assets: int = 10,
) -> AtomicPeriodScore:
    """Score one period using the challenge's eight atomic metrics.

    Predictions and meta predictions are uniform-ranked before scoring, which
    mirrors the production challenge pipeline. Targets must already be ranking
    labels in ``[0, 1]``.
    """

    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (y_true_10d, y_pred_10d, y_true_30d, y_pred_30d)
    )
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError("target and prediction arrays must have equal lengths")
    n_assets = lengths.pop()
    if n_assets < min_assets:
        raise InsufficientAssetsError(
            f"period has {n_assets} assets; at least {min_assets} are required"
        )
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("target and prediction arrays must be one-dimensional")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("target and prediction arrays must be finite")

    true_10d, pred_10d, true_30d, pred_30d = arrays
    ranked_pred_10d = uniform_rank(pred_10d)
    ranked_pred_30d = uniform_rank(pred_30d)
    raw = evaluate_hyperliquid_submission(
        true_10d,
        ranked_pred_10d,
        true_30d,
        ranked_pred_30d,
    )

    if (meta_pred_10d is None) != (meta_pred_30d is None):
        raise ValueError("both meta horizons must be supplied together")
    if meta_pred_10d is None:
        return AtomicPeriodScore(raw=raw)

    meta_10d = np.asarray(meta_pred_10d, dtype=float)
    meta_30d = np.asarray(meta_pred_30d, dtype=float)
    if meta_10d.shape != pred_10d.shape or meta_30d.shape != pred_30d.shape:
        raise ValueError("meta predictions must match prediction shapes")
    unique = evaluate_hyperliquid_uniqueness(
        true_10d,
        ranked_pred_10d,
        uniform_rank(meta_10d),
        true_30d,
        ranked_pred_30d,
        uniform_rank(meta_30d),
    )
    return AtomicPeriodScore(raw=raw, unique=unique)


def aggregate_objective(
    scores: list[AtomicPeriodScore],
    objective: PointsObjective,
) -> float:
    """Mean score-space objective across out-of-sample periods."""

    if not scores:
        raise ValueError("at least one period score is required")
    return fmean(score.objective_proxy(objective) for score in scores)


def book_objective(
    slot_scores: Mapping[int, AtomicPeriodScore],
    objective: PointsObjective,
    *,
    scored_slots: set[int] | None = None,
) -> float:
    """Equal-weight score-space objective across non-experimental slots."""

    selected = set(slot_scores) if scored_slots is None else set(scored_slots)
    if not selected:
        raise ValueError("at least one scored slot is required")
    missing = selected.difference(slot_scores)
    if missing:
        raise ValueError(f"missing scores for slots: {sorted(missing)}")
    return fmean(slot_scores[slot].objective_proxy(objective) for slot in selected)
