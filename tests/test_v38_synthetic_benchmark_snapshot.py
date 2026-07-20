"""Frozen SYNTHETIC v3.8 pipeline milestone — not a live HL result."""

import json
from datetime import date
from pathlib import Path

import pytest
import xgboost
from crowdcent_challenge.validation.v38 import V38RunConfig, run_v38_smoke

SNAPSHOT = Path(__file__).parent / "data" / "v38_synthetic_benchmark.json"


def test_v38_synthetic_benchmark_snapshot():
    golden = json.loads(SNAPSHOT.read_text())

    assert xgboost.__version__ == golden["env"]["xgboost"], (
        "xgboost changed; rerun run_v38_smoke(seed=11) and regenerate "
        "tests/data/v38_synthetic_benchmark.json deliberately."
    )

    cfg = golden["config"]
    result = run_v38_smoke(
        V38RunConfig(
            assets_per_period=cfg["assets_per_period"],
            n_features=cfg["n_features"],
            seed=cfg["seed"],
        )
    )

    assert list(result.selected_candidate_ids) == golden["selected_candidate_ids"]
    assert result.outer_validation_start == date.fromisoformat(golden["outer_validation_start"])
    assert result.outer_validation_end == date.fromisoformat(golden["outer_validation_end"])
    assert result.outer_periods == golden["outer_periods"]
    assert result.inner_periods == golden["inner_periods"]
    assert len(result.inner_selection.evaluations) == golden["n_evaluations"]

    assert len(result.selected_candidate_ids) <= 5
    assert result.outer_delta > 5.0
    assert result.outer_delta_ci[0] > 0.0
    assert result.outer_v38_points > result.outer_v36_points

    tol = golden["tolerance_points_abs"]
    assert result.outer_v38_points == pytest.approx(golden["outer_v38_points"], abs=tol)
    assert result.outer_v36_points == pytest.approx(golden["outer_v36_points"], abs=tol)
    assert result.outer_delta == pytest.approx(golden["outer_delta"], abs=tol)
