from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from crowdcent_challenge.validation.v38 import (
    DEPLOYMENT_CANDIDATES,
    PredictionPanel,
    build_live_meta_input,
    build_v38_deployment_book,
    deployment_audit,
    submit_v38_deployment_book,
)


def _components(period: date, count: int = 12) -> dict[str, PredictionPanel]:
    ids = tuple(f"asset-{index}" for index in range(count))
    periods = tuple(period for _ in ids)
    xgb = np.column_stack(
        [
            np.linspace(0.1, 0.9, count),
            np.linspace(0.9, 0.1, count),
        ]
    )
    transformer = np.column_stack(
        [
            np.roll(np.linspace(0.1, 0.9, count), 2),
            np.roll(np.linspace(0.9, 0.1, count), 3),
        ]
    )
    return {
        "xgb": PredictionPanel(periods=periods, ids=ids, values=xgb),
        "transformer": PredictionPanel(periods=periods, ids=ids, values=transformer),
    }


def _meta_frame(period: date, count: int = 12) -> pl.DataFrame:
    prior = period.replace(day=period.day - 1)
    rows = []
    for release in (prior, period):
        for index in range(count):
            rows.append(
                {
                    "id": f"asset-{index}",
                    "release_date": release,
                    "pred_10d": (index + 0.5) / count,
                    "pred_30d": (count - index - 0.5) / count,
                }
            )
    return pl.DataFrame(rows)


def test_live_meta_input_uses_strictly_prior_release() -> None:
    period = date(2026, 7, 18)
    reference = _components(period)["transformer"]

    meta_input = build_live_meta_input(reference, _meta_frame(period))

    assert set(meta_input.release_dates) == {date(2026, 7, 17)}
    assert all(
        release < prediction_period
        for release, prediction_period in zip(
            meta_input.release_dates,
            meta_input.predictions.periods,
            strict=True,
        )
    )


def test_live_meta_input_rejects_stale_cache() -> None:
    period = date(2026, 7, 18)
    reference = _components(period)["transformer"]
    stale = (
        _meta_frame(period)
        .filter(pl.col("release_date") == period - timedelta(days=1))
        .with_columns(pl.lit(period - timedelta(days=8)).alias("release_date"))
    )

    with pytest.raises(ValueError, match="stale"):
        build_live_meta_input(reference, stale)


def test_deployment_book_has_one_scored_and_four_experimental_slots() -> None:
    period = date(2026, 7, 18)

    slots = build_v38_deployment_book(_components(period), _meta_frame(period))
    audit = deployment_audit(slots)

    assert tuple(slot.slot for slot in slots) == (1, 2, 3, 4, 5)
    assert tuple(slot.candidate for slot in slots) == DEPLOYMENT_CANDIDATES
    assert [slot.is_experimental for slot in slots] == [False, True, True, True, True]
    assert audit["scored_slots"] == 1
    assert audit["experimental_slots"] == 4
    assert len(audit["pairwise_rank_correlations"]) == 10


def test_deployment_submit_is_dry_by_default_and_scored_first_when_enabled() -> None:
    period = date(2026, 7, 18)
    slots = build_v38_deployment_book(_components(period), _meta_frame(period))

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def submit_predictions(
            self,
            *,
            df: pl.DataFrame,
            slot: int,
            is_experimental: bool,
            notes: str,
        ) -> dict[str, object]:
            assert df.columns == ["id", "pred_10d", "pred_30d"]
            assert notes.startswith("v3.8 ")
            self.calls.append((slot, is_experimental))
            return {"slot": slot}

    client = FakeClient()
    dry_run = submit_v38_deployment_book(client, slots)
    assert client.calls == []
    assert all(result["status"] == "dry_run" for result in dry_run.values())

    submit_v38_deployment_book(client, slots, submit=True)
    assert client.calls == [
        (1, False),
        (2, True),
        (3, True),
        (4, True),
        (5, True),
    ]
