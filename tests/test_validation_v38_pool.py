"""Tournament pool calibration tests."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from crowdcent_challenge.validation.atomic import META_DIAGNOSTICS, RAW_METRICS, UNIQUE_METRICS
from crowdcent_challenge.validation.v38 import (
    ComponentFitConfig,
    PoolReferenceConfig,
    PoolSource,
    V38NestedFoldConfig,
    V38RunConfig,
    build_pool_reference,
    candidate_library,
    load_cached_pool_scores,
    make_synthetic_dataset,
    run_v38_smoke,
    run_v38_walkforward_backtest,
    score_meta_as_participant,
)


def _atomic_row(model_id: str, period: date, value: float) -> dict:
    return {
        "model_id": model_id,
        "period": period,
        **{metric: value for metric in RAW_METRICS},
        **{metric: value for metric in UNIQUE_METRICS},
    }


def test_own_candidates_source_matches_legacy_smoke():
    baseline = run_v38_smoke(V38RunConfig(assets_per_period=24, n_features=12, seed=11))
    explicit = run_v38_smoke(
        V38RunConfig(
            assets_per_period=24,
            n_features=12,
            seed=11,
            pool_reference=PoolReferenceConfig(
                source=PoolSource.OWN_CANDIDATES,
                include_incumbent_slots=True,
            ),
        )
    )
    assert explicit.selected_candidate_ids == baseline.selected_candidate_ids
    assert explicit.outer_delta == pytest.approx(baseline.outer_delta)


def test_score_meta_as_participant_skips_first_meta_period():
    start = date(2025, 6, 5)
    periods = [start + timedelta(days=offset) for offset in range(5)]
    frame = pl.DataFrame(
        {
            "date": [period for period in periods for _ in range(12)],
            "id": [f"A{i % 12:02d}" for i in range(len(periods) * 12)],
            "target_10d": [0.5] * (len(periods) * 12),
            "target_30d": [0.5] * (len(periods) * 12),
            "feature_0": [0.1] * (len(periods) * 12),
        }
    )
    from crowdcent_challenge.validation.v38 import prepared_dataset_from_frame

    dataset = prepared_dataset_from_frame(frame)
    meta = pl.DataFrame(
        {
            "release_date": periods,
            "id": ["A00"] * len(periods),
            "pred_10d": [0.4, 0.41, 0.42, 0.43, 0.44],
            "pred_30d": [0.6, 0.61, 0.62, 0.63, 0.64],
        }
    )
    scores = score_meta_as_participant(dataset, meta, min_assets=10)
    assert scores
    assert all(score.unique is not None for score in scores)


def test_load_cached_pool_scores_schema_and_pit_filter(tmp_path: Path):
    inside = date(2026, 1, 1)
    outside = date(2026, 2, 1)
    frame = pl.DataFrame(
        [_atomic_row("m1", inside, 0.5), _atomic_row("m2", outside, 0.8)]
    )
    path = tmp_path / "pool.parquet"
    frame.write_parquet(path)
    rows = load_cached_pool_scores(path, reference_periods={inside})
    assert len(rows) == 1
    with pytest.raises(FileNotFoundError):
        load_cached_pool_scores(tmp_path / "missing.parquet", reference_periods={inside})


def test_pool_fallback_when_cached_missing():
    from crowdcent_challenge.validation.atomic import AtomicPeriodScore
    from crowdcent_challenge.validation.v38.objective import PeriodScore

    score = AtomicPeriodScore(
        raw={metric: 0.5 for metric in RAW_METRICS},
        unique={
            **{metric: 0.5 for metric in UNIQUE_METRICS},
            **{metric: 0.0 for metric in META_DIAGNOSTICS},
        },
    )
    inner_atomic = {
        "c1": [PeriodScore(period=date(2026, 1, 1), score=score)],
    }
    pool = build_pool_reference(
        config=PoolReferenceConfig(
            source=PoolSource.CACHED,
            cached_pool_path=Path("/nonexistent/pool.parquet"),
            min_reference_models=9999,
        ),
        inner_atomic=inner_atomic,
    )
    assert pool.source is PoolSource.OWN_CANDIDATES
    assert "fallback" in pool.provenance


def test_walkforward_pool_calibrated_smoke():
    dataset = make_synthetic_dataset(V38RunConfig(assets_per_period=20, n_features=8, seed=3))
    result = run_v38_walkforward_backtest(
        dataset,
        config=V38RunConfig(
            assets_per_period=20,
            n_features=8,
            seed=3,
            fold_config=V38NestedFoldConfig(
                outer_min_train_days=360,
                outer_validation_days=60,
                outer_step_days=60,
                outer_max_folds=1,
                inner_min_train_days=180,
                inner_validation_days=45,
                inner_step_days=45,
                inner_folds=1,
                embargo_days=30,
            ),
            component_fit=ComponentFitConfig(mode="smoke", seed=3),
            pool_reference=PoolReferenceConfig(source=PoolSource.META_ANCHORED),
        ),
        specs=candidate_library(smoke=True),
        max_folds=1,
    )
    assert result.folds[0].result.pool_provenance
