"""v3.6 five-slot incumbent baseline — comparison only, not v3.8 selection."""

from collections.abc import Mapping

import numpy as np

from crowdcent_challenge.validation.training.recipe import (
    HorizonRecipe,
    RawHorizonPredictions,
    StackWeights,
    build_five_slot_predictions,
)

from .contracts import MetaInputPanel, PredictionPanel

# INCUMBENT BASELINE ONLY — not part of v3.8 candidate selection.
V36_RECIPE = HorizonRecipe(
    slot1_meta_weight=0.40,
    slot2_meta_weight=0.40,
    slot3_slot1_weight=0.35,
    slot4_slot1_weight=0.50,
    slot4_meta_weight=0.25,
    slot5=StackWeights(
        xgb=0.18,
        lstm=0.15,
        transformer=0.20,
        bilstm=0.14,
        meta=0.33,
    ),
)
V36_SLOT_IDS: tuple[str, ...] = tuple(f"v36:slot{slot}" for slot in range(1, 6))


def build_incumbent_baseline_panels(
    components: Mapping[str, PredictionPanel],
    meta_input: MetaInputPanel,
) -> dict[str, PredictionPanel]:
    """Materialize the fixed v3.6 five-slot book for paired comparison."""

    reference = components["xgb"]
    slot_values = {
        slot_id: np.empty((len(reference.periods), 2), dtype=float) for slot_id in V36_SLOT_IDS
    }
    period_array = np.asarray(reference.periods, dtype=object)
    meta_lookup = dict(zip(meta_input.predictions.keys, meta_input.predictions.values, strict=True))
    for period in sorted(set(reference.periods)):
        mask = period_array == period
        for horizon in range(2):
            raw = RawHorizonPredictions(
                xgb=components["xgb"].values[mask, horizon],
                lstm=components["lstm"].values[mask, horizon],
                transformer=components["transformer"].values[mask, horizon],
                bilstm=components["bilstm"].values[mask, horizon],
                meta=np.asarray(
                    [
                        meta_lookup[(period, asset_id)][horizon]
                        for asset_id, keep in zip(reference.ids, mask, strict=True)
                        if keep
                    ]
                ),
            )
            slots = build_five_slot_predictions(raw, V36_RECIPE)
            for slot_index, values in enumerate(
                (slots.slot1, slots.slot2, slots.slot3, slots.slot4, slots.slot5),
                start=1,
            ):
                slot_values[f"v36:slot{slot_index}"][mask, horizon] = values
    return {
        slot_id: PredictionPanel(
            periods=reference.periods,
            ids=reference.ids,
            values=values,
        )
        for slot_id, values in slot_values.items()
    }
