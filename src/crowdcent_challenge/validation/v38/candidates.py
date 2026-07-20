"""Deterministic candidate construction without fixed slot identities."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from crowdcent_challenge.scoring import neutralize_predictions

from .contracts import MetaInputPanel, PredictionPanel, require_same_keys


def _rank_within_period(panel: PredictionPanel) -> PredictionPanel:
    values = np.empty_like(panel.values, dtype=float)
    period_values = np.asarray(panel.periods, dtype=object)
    for period in sorted(set(panel.periods)):
        indices = np.flatnonzero(period_values == period)
        for horizon in range(2):
            source = panel.values[indices, horizon]
            values[indices, horizon] = (rankdata(source, method="average") - 0.5) / len(source)
    return PredictionPanel(periods=panel.periods, ids=panel.ids, values=values)


@dataclass(frozen=True)
class PrimitiveCandidate:
    component_id: Hashable

    @property
    def candidate_id(self) -> str:
        return f"primitive:{self.component_id}"


@dataclass(frozen=True)
class BlendCandidate:
    component_ids: tuple[Hashable, ...]
    weights_10d: tuple[float, ...]
    weights_30d: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.component_ids)
        if count < 2 or len(set(self.component_ids)) != count:
            raise ValueError("blend candidates require distinct components")
        for weights in (self.weights_10d, self.weights_30d):
            values = np.asarray(weights, dtype=float)
            if len(values) != count:
                raise ValueError("blend weights must align with components")
            if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError("blend weights must be finite, non-negative, and non-zero")

    @property
    def candidate_id(self) -> str:
        components = ",".join(map(str, self.component_ids))
        ten = ",".join(f"{weight:.3f}" for weight in self.weights_10d)
        thirty = ",".join(f"{weight:.3f}" for weight in self.weights_30d)
        return f"blend:{components}:10={ten}:30={thirty}"


BaseCandidate = PrimitiveCandidate | BlendCandidate


@dataclass(frozen=True)
class PriorMetaBlendCandidate:
    base: BaseCandidate
    meta_weight_10d: float
    meta_weight_30d: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.meta_weight_10d <= 1.0:
            raise ValueError("meta_weight_10d must be in [0, 1]")
        if not 0.0 <= self.meta_weight_30d <= 1.0:
            raise ValueError("meta_weight_30d must be in [0, 1]")

    @property
    def candidate_id(self) -> str:
        return (
            f"prior-meta:{self.base.candidate_id}:"
            f"{self.meta_weight_10d:.3f},{self.meta_weight_30d:.3f}"
        )


@dataclass(frozen=True)
class ResidualCandidate:
    base: BaseCandidate
    strength_10d: float
    strength_30d: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength_10d <= 1.0:
            raise ValueError("strength_10d must be in [0, 1]")
        if not 0.0 <= self.strength_30d <= 1.0:
            raise ValueError("strength_30d must be in [0, 1]")

    @property
    def candidate_id(self) -> str:
        return f"residual:{self.base.candidate_id}:{self.strength_10d:.3f},{self.strength_30d:.3f}"


CandidateSpec = PrimitiveCandidate | BlendCandidate | PriorMetaBlendCandidate | ResidualCandidate


def build_candidate_panel(
    candidate: CandidateSpec,
    components: Mapping[Hashable, PredictionPanel],
    *,
    meta_input: MetaInputPanel | None = None,
) -> PredictionPanel:
    """Evaluate a declared candidate using inference-feasible inputs only."""

    if isinstance(candidate, PrimitiveCandidate):
        if candidate.component_id not in components:
            raise ValueError(f"unknown component: {candidate.component_id}")
        return _rank_within_period(components[candidate.component_id])

    if isinstance(candidate, BlendCandidate):
        panels = [components[component_id] for component_id in candidate.component_ids]
        require_same_keys(*panels)
        values = np.empty_like(panels[0].values, dtype=float)
        for horizon, raw_weights in enumerate((candidate.weights_10d, candidate.weights_30d)):
            weights = np.asarray(raw_weights, dtype=float)
            weights /= weights.sum()
            values[:, horizon] = sum(
                weight * _rank_within_period(panel).values[:, horizon]
                for panel, weight in zip(panels, weights, strict=True)
            )
        return _rank_within_period(
            PredictionPanel(
                periods=panels[0].periods,
                ids=panels[0].ids,
                values=values,
            )
        )

    if meta_input is None:
        raise ValueError("prior-meta candidates require a MetaInputPanel")
    if not isinstance(meta_input, MetaInputPanel):
        raise TypeError("candidate construction accepts MetaInputPanel only")
    base = build_candidate_panel(candidate.base, components)
    prior_meta = _rank_within_period(meta_input.predictions)
    require_same_keys(base, prior_meta)
    values = np.empty_like(base.values)

    if isinstance(candidate, PriorMetaBlendCandidate):
        for horizon, meta_weight in enumerate(
            (candidate.meta_weight_10d, candidate.meta_weight_30d)
        ):
            values[:, horizon] = (1.0 - meta_weight) * base.values[
                :, horizon
            ] + meta_weight * prior_meta.values[:, horizon]
    else:
        period_values = np.asarray(base.periods, dtype=object)
        for period in sorted(set(base.periods)):
            indices = np.flatnonzero(period_values == period)
            for horizon, strength in enumerate((candidate.strength_10d, candidate.strength_30d)):
                source = base.values[indices, horizon]
                residual = neutralize_predictions(
                    source,
                    prior_meta.values[indices, horizon],
                )
                values[indices, horizon] = (1.0 - strength) * source + strength * residual
    return _rank_within_period(PredictionPanel(periods=base.periods, ids=base.ids, values=values))


def generate_sparse_blends(
    component_ids: tuple[Hashable, ...],
    *,
    pair_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[BaseCandidate]:
    """Predeclare primitive and sparse pair candidates deterministically."""

    if len(set(component_ids)) != len(component_ids):
        raise ValueError("component_ids must be unique")
    candidates: list[BaseCandidate] = [
        PrimitiveCandidate(component_id) for component_id in component_ids
    ]
    for left, right in combinations(component_ids, 2):
        for weight in pair_weights:
            if not 0.0 < weight < 1.0:
                raise ValueError("pair weights must be strictly between zero and one")
            weights = (float(weight), float(1.0 - weight))
            candidates.append(
                BlendCandidate(
                    component_ids=(left, right),
                    weights_10d=weights,
                    weights_30d=weights,
                )
            )
    return candidates
