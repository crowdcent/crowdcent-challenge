"""Fixed, audited live book for the validated v3.8 deployment candidate."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from crowdcent_challenge.validation.pit import MetaPurpose, latest_available_meta_release

from .candidates import (
    BlendCandidate,
    CandidateSpec,
    PrimitiveCandidate,
    PriorMetaBlendCandidate,
    build_candidate_panel,
)
from .contracts import MetaInputPanel, PredictionPanel, require_same_keys

SCORED_CANDIDATE = PriorMetaBlendCandidate(
    base=PrimitiveCandidate("transformer"),
    meta_weight_10d=0.25,
    meta_weight_30d=0.25,
)
EXPERIMENTAL_CANDIDATES: tuple[CandidateSpec, ...] = (
    BlendCandidate(
        component_ids=("xgb", "transformer"),
        weights_10d=(0.5, 0.5),
        weights_30d=(0.5, 0.5),
    ),
    PrimitiveCandidate("transformer"),
    PriorMetaBlendCandidate(
        base=PrimitiveCandidate("xgb"),
        meta_weight_10d=0.25,
        meta_weight_30d=0.25,
    ),
    BlendCandidate(
        component_ids=("xgb", "transformer"),
        weights_10d=(0.25, 0.75),
        weights_30d=(0.25, 0.75),
    ),
)
DEPLOYMENT_CANDIDATES: tuple[CandidateSpec, ...] = (
    SCORED_CANDIDATE,
    *EXPERIMENTAL_CANDIDATES,
)


@dataclass(frozen=True)
class DeploymentSlot:
    """One API slot and its immutable v3.8 candidate recipe."""

    slot: int
    candidate: CandidateSpec
    panel: PredictionPanel
    is_experimental: bool
    meta_release_dates: tuple[date, ...]

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id


def build_live_meta_input(
    reference: PredictionPanel,
    meta_frame: Any,
    *,
    model_input_lag_days: int = 1,
    max_meta_age_days: int = 7,
) -> MetaInputPanel:
    """Align the latest admissible prior meta snapshot to live prediction keys."""

    if max_meta_age_days < 1:
        raise ValueError("max_meta_age_days must be positive")
    pl = _import_polars()
    required = {"id", "release_date", "pred_10d", "pred_30d"}
    missing = required - set(meta_frame.columns)
    if missing:
        raise ValueError(f"meta frame is missing columns: {sorted(missing)}")
    meta = meta_frame.with_columns(pl.col("release_date").cast(pl.Date))
    release_dates = tuple(sorted(meta["release_date"].unique().to_list()))
    releases_by_period: dict[date, date] = {}
    lookups: dict[date, dict[Hashable, tuple[float, float]]] = {}
    for period in sorted(set(reference.periods)):
        release = latest_available_meta_release(
            release_dates,
            period,
            MetaPurpose.MODEL_INPUT,
            model_input_lag_days=model_input_lag_days,
        )
        if release is None:
            raise ValueError(f"no admissible prior meta release exists for {period}")
        if release >= period:
            raise ValueError(f"meta release {release} is not strictly earlier than {period}")
        if (period - release).days > max_meta_age_days:
            raise ValueError(
                f"latest prior meta release {release} is stale for {period} "
                f"(maximum age: {max_meta_age_days} days)"
            )
        releases_by_period[period] = release
        subset = meta.filter(pl.col("release_date") == release)
        lookups[period] = {
            row["id"]: (float(row["pred_10d"]), float(row["pred_30d"]))
            for row in subset.select(["id", "pred_10d", "pred_30d"]).iter_rows(named=True)
        }

    values = np.asarray(
        [
            lookups[period].get(asset_id, (0.5, 0.5))
            for period, asset_id in zip(reference.periods, reference.ids, strict=True)
        ],
        dtype=float,
    )
    return MetaInputPanel(
        predictions=PredictionPanel(
            periods=reference.periods,
            ids=reference.ids,
            values=values,
        ),
        release_dates=tuple(releases_by_period[period] for period in reference.periods),
    )


def build_v38_deployment_book(
    components: Mapping[Hashable, PredictionPanel],
    meta_frame: Any,
) -> tuple[DeploymentSlot, ...]:
    """Build one scored slot and four shadow slots from fixed candidate specs."""

    missing = {"xgb", "transformer"} - set(components)
    if missing:
        raise ValueError(f"deployment components are missing: {sorted(missing)}")
    reference = components["transformer"]
    require_same_keys(reference, components["xgb"])
    meta_input = build_live_meta_input(reference, meta_frame)
    slots = tuple(
        DeploymentSlot(
            slot=index,
            candidate=candidate,
            panel=build_candidate_panel(candidate, components, meta_input=meta_input),
            is_experimental=index != 1,
            meta_release_dates=meta_input.release_dates,
        )
        for index, candidate in enumerate(DEPLOYMENT_CANDIDATES, start=1)
    )
    validate_v38_deployment_book(slots)
    return slots


def validate_v38_deployment_book(slots: tuple[DeploymentSlot, ...]) -> None:
    """Fail closed on role, alignment, range, or recipe drift."""

    if tuple(slot.slot for slot in slots) != (1, 2, 3, 4, 5):
        raise ValueError("v3.8 deployment requires exactly API slots 1 through 5")
    if tuple(slot.candidate for slot in slots) != DEPLOYMENT_CANDIDATES:
        raise ValueError("v3.8 deployment candidate recipes have drifted")
    if sum(not slot.is_experimental for slot in slots) != 1:
        raise ValueError("v3.8 deployment requires exactly one scored slot")
    if slots[0].is_experimental or any(not slot.is_experimental for slot in slots[1:]):
        raise ValueError("slot 1 must be scored and slots 2 through 5 experimental")
    require_same_keys(*(slot.panel for slot in slots))
    if any(slot.meta_release_dates != slots[0].meta_release_dates for slot in slots[1:]):
        raise ValueError("deployment slots do not share identical prior meta releases")
    if any(
        release >= period
        for release, period in zip(
            slots[0].meta_release_dates,
            slots[0].panel.periods,
            strict=True,
        )
    ):
        raise ValueError("deployment contains a non-point-in-time meta release")
    if len(slots[0].panel.ids) < 10:
        raise ValueError("deployment cross-section must contain at least 10 assets")
    for slot in slots:
        if not np.isfinite(slot.panel.values).all():
            raise ValueError(f"slot {slot.slot} contains non-finite predictions")
        if (slot.panel.values <= 0.0).any() or (slot.panel.values >= 1.0).any():
            raise ValueError(f"slot {slot.slot} predictions must be strictly inside (0, 1)")


def deployment_audit(slots: tuple[DeploymentSlot, ...]) -> dict[str, Any]:
    """Return a compact, serializable pre-submission decomposition report."""

    validate_v38_deployment_book(slots)
    rows = []
    for slot in slots:
        rows.append(
            {
                "slot": slot.slot,
                "role": "experimental" if slot.is_experimental else "scored",
                "candidate_id": slot.candidate_id,
                "assets": len(slot.panel.ids),
                "periods": [str(period) for period in sorted(set(slot.panel.periods))],
                "pred_10d_range": [
                    float(slot.panel.values[:, 0].min()),
                    float(slot.panel.values[:, 0].max()),
                ],
                "pred_30d_range": [
                    float(slot.panel.values[:, 1].min()),
                    float(slot.panel.values[:, 1].max()),
                ],
            }
        )
    correlations = []
    for left_index, left in enumerate(slots):
        for right in slots[left_index + 1 :]:
            correlations.append(
                {
                    "left_slot": left.slot,
                    "right_slot": right.slot,
                    "spearman_10d": float(
                        np.corrcoef(left.panel.values[:, 0], right.panel.values[:, 0])[0, 1]
                    ),
                    "spearman_30d": float(
                        np.corrcoef(left.panel.values[:, 1], right.panel.values[:, 1])[0, 1]
                    ),
                }
            )
    return {
        "version": "3.8",
        "scored_slots": 1,
        "experimental_slots": 4,
        "meta_release_dates": [
            str(release) for release in sorted(set(slots[0].meta_release_dates))
        ],
        "slots": rows,
        "pairwise_rank_correlations": correlations,
    }


def panel_to_submission_frame(panel: PredictionPanel) -> Any:
    """Convert an aligned prediction panel to the Challenge API schema."""

    pl = _import_polars()
    return pl.DataFrame(
        {
            "id": panel.ids,
            "pred_10d": panel.values[:, 0],
            "pred_30d": panel.values[:, 1],
        }
    )


def submit_v38_deployment_book(
    client: Any,
    slots: tuple[DeploymentSlot, ...],
    *,
    submit: bool = False,
) -> dict[int, Any]:
    """Submit scored-first, or return an explicit no-side-effect dry run."""

    validate_v38_deployment_book(slots)
    if not submit:
        return {
            slot.slot: {
                "status": "dry_run",
                "candidate_id": slot.candidate_id,
                "is_experimental": slot.is_experimental,
            }
            for slot in slots
        }
    results = {}
    for slot in slots:
        results[slot.slot] = client.submit_predictions(
            df=panel_to_submission_frame(slot.panel),
            slot=slot.slot,
            is_experimental=slot.is_experimental,
            notes=f"v3.8 {slot.candidate_id}",
        )
    return results


def _import_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "v3.8 deployment requires polars; install validation-training extra"
        ) from exc
    return pl
