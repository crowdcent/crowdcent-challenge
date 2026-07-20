"""Pure rank-space construction of the five v3.6 model slots."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def midpoint_rank(values: np.ndarray) -> np.ndarray:
    """Notebook-compatible uniform ranks in ``(0, 1)``."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return (rankdata(array, method="average") - 0.5) / len(array)


def rank_blend(*components: tuple[np.ndarray, float]) -> np.ndarray:
    """Weight signals in rank space and re-rank the mixture."""

    if not components:
        raise ValueError("at least one component is required")
    weights = np.asarray([weight for _, weight in components], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("component weights must be finite, non-negative, and non-zero")
    arrays = [np.asarray(values, dtype=float) for values, _ in components]
    shapes = {values.shape for values in arrays}
    if len(shapes) != 1 or arrays[0].ndim != 1:
        raise ValueError("component arrays must be aligned one-dimensional arrays")
    mixture = sum(
        weight / weights.sum() * midpoint_rank(values)
        for values, weight in zip(arrays, weights, strict=True)
    )
    return midpoint_rank(mixture)


@dataclass(frozen=True)
class StackWeights:
    xgb: float
    lstm: float
    transformer: float
    bilstm: float
    meta: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.xgb, self.lstm, self.transformer, self.bilstm, self.meta],
            dtype=float,
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("stack weights must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("stack weights must sum to one")
        if self.meta > 0.33 + 1e-12:
            raise ValueError("meta weight exceeds the v3.6 cap")
        if self.xgb > 0.22 + 1e-12:
            raise ValueError("xgb weight exceeds the v3.6 cap")


@dataclass(frozen=True)
class HorizonRecipe:
    slot1_meta_weight: float
    slot2_meta_weight: float
    slot3_slot1_weight: float
    slot4_slot1_weight: float
    slot4_meta_weight: float
    slot5: StackWeights

    def __post_init__(self) -> None:
        scalar_weights = (
            self.slot1_meta_weight,
            self.slot2_meta_weight,
            self.slot3_slot1_weight,
            self.slot4_slot1_weight,
            self.slot4_meta_weight,
        )
        if any(not 0.0 <= weight <= 1.0 for weight in scalar_weights):
            raise ValueError("recipe weights must be in [0, 1]")


@dataclass(frozen=True)
class RawHorizonPredictions:
    """Aligned predictions emitted by the four fitted model families."""

    xgb: np.ndarray
    lstm: np.ndarray
    transformer: np.ndarray
    bilstm: np.ndarray
    meta: np.ndarray

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(getattr(self, name), dtype=float)
            for name in ("xgb", "lstm", "transformer", "bilstm", "meta")
        ]
        shapes = {values.shape for values in arrays}
        if len(shapes) != 1 or arrays[0].ndim != 1:
            raise ValueError("raw model predictions must be aligned 1D arrays")
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("raw model predictions must be finite")


@dataclass(frozen=True)
class FiveSlotPredictions:
    slot1: np.ndarray
    slot2: np.ndarray
    slot3: np.ndarray
    slot4: np.ndarray
    slot5: np.ndarray

    def as_dict(self) -> dict[int, np.ndarray]:
        return {
            1: self.slot1,
            2: self.slot2,
            3: self.slot3,
            4: self.slot4,
            5: self.slot5,
        }


def build_five_slot_predictions(
    raw: RawHorizonPredictions,
    recipe: HorizonRecipe,
) -> FiveSlotPredictions:
    """Construct all five notebook slots from real model-family predictions."""

    slot1 = rank_blend(
        (raw.meta, recipe.slot1_meta_weight),
        (raw.xgb, 1.0 - recipe.slot1_meta_weight),
    )
    slot2 = rank_blend(
        (raw.meta, recipe.slot2_meta_weight),
        (raw.transformer, 1.0 - recipe.slot2_meta_weight),
    )
    slot3 = rank_blend(
        (raw.xgb, recipe.slot3_slot1_weight),
        (raw.transformer, 1.0 - recipe.slot3_slot1_weight),
    )
    slot4_alpha = rank_blend(
        (raw.xgb, recipe.slot4_slot1_weight),
        (raw.transformer, 1.0 - recipe.slot4_slot1_weight),
    )
    slot4 = rank_blend(
        (raw.meta, recipe.slot4_meta_weight),
        (slot4_alpha, 1.0 - recipe.slot4_meta_weight),
    )
    slot5 = rank_blend(
        (raw.xgb, recipe.slot5.xgb),
        (raw.lstm, recipe.slot5.lstm),
        (raw.transformer, recipe.slot5.transformer),
        (raw.bilstm, recipe.slot5.bilstm),
        (raw.meta, recipe.slot5.meta),
    )
    return FiveSlotPredictions(
        slot1=slot1,
        slot2=slot2,
        slot3=slot3,
        slot4=slot4,
        slot5=slot5,
    )
