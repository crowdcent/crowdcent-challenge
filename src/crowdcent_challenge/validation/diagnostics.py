"""Diagnostics for universe drift and meta-model self-inclusion."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np

from crowdcent_challenge.scoring import spearman_correlation


@dataclass(frozen=True)
class UniverseChurn:
    added: frozenset[Hashable]
    removed: frozenset[Hashable]
    retained: frozenset[Hashable]
    jaccard_similarity: float


def universe_churn(
    previous: set[Hashable],
    current: set[Hashable],
) -> UniverseChurn:
    """Describe point-in-time asset-universe changes."""

    union = previous | current
    similarity = len(previous & current) / len(union) if union else 1.0
    return UniverseChurn(
        added=frozenset(current - previous),
        removed=frozenset(previous - current),
        retained=frozenset(previous & current),
        jaccard_similarity=similarity,
    )


def weighted_meta_model(
    user_predictions: Mapping[str, np.ndarray],
    user_weights: Mapping[str, float],
    *,
    exclude_user: str | None = None,
) -> np.ndarray:
    """Build a weighted user-level meta-model, optionally leave-one-user-out.

    Inputs must already be user-level averages of uniform-ranked,
    non-experimental slots. This mirrors the production aggregation boundary.
    """

    included = {
        user: np.asarray(predictions, dtype=float)
        for user, predictions in user_predictions.items()
        if user != exclude_user
    }
    if not included:
        raise ValueError("at least one user must remain in the meta-model")
    shapes = {predictions.shape for predictions in included.values()}
    if len(shapes) != 1:
        raise ValueError("all user prediction arrays must have the same shape")
    if any(predictions.ndim != 1 for predictions in included.values()):
        raise ValueError("user prediction arrays must be one-dimensional")

    weights = np.array([user_weights.get(user, 0.0) for user in included], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("user weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("included user weights must sum to a positive value")
    matrix = np.stack(list(included.values()))
    if not np.isfinite(matrix).all():
        raise ValueError("user predictions must be finite")
    return np.average(matrix, axis=0, weights=weights)


@dataclass(frozen=True)
class SelfInclusionDiagnostic:
    full_vs_leave_one_out_spearman: float
    mean_absolute_shift: float
    max_absolute_shift: float


def self_inclusion_diagnostic(
    user: str,
    user_predictions: Mapping[str, np.ndarray],
    user_weights: Mapping[str, float],
) -> SelfInclusionDiagnostic:
    """Measure how much one user's signal moves the scoring meta-model."""

    if user not in user_predictions:
        raise ValueError(f"unknown user: {user}")
    full = weighted_meta_model(user_predictions, user_weights)
    leave_one_out = weighted_meta_model(
        user_predictions,
        user_weights,
        exclude_user=user,
    )
    shift = np.abs(full - leave_one_out)
    return SelfInclusionDiagnostic(
        full_vs_leave_one_out_spearman=spearman_correlation(full, leave_one_out),
        mean_absolute_shift=float(np.mean(shift)),
        max_absolute_shift=float(np.max(shift)),
    )
