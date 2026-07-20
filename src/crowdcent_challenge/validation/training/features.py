"""Feature preparation extracted from the v3.6 notebook."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._optional import import_training_dependency

FEATURE_RE = re.compile(r"^feature_(\d+)_lag(\d+)$")
DEFAULT_LAGS = (0, 5, 10, 15)
DIFF_PAIRS = ((0, 5), (5, 10), (10, 15), (0, 15))


@dataclass(frozen=True)
class SequenceFeatureSchema:
    """Validated lag-major sequence feature layout."""

    ordered_columns: tuple[str, ...]
    feature_ids: tuple[int, ...]
    lag_windows: tuple[int, ...]

    @property
    def n_features_per_timestep(self) -> int:
        return len(self.feature_ids)


def sequence_feature_schema(
    columns: Iterable[str],
    *,
    expected_lags: tuple[int, ...] | None = DEFAULT_LAGS,
) -> SequenceFeatureSchema:
    """Parse and validate a complete feature-by-lag grid.

    Centimators reshapes a flat matrix by lag-major blocks. Missing one lag
    silently shifts every later feature, so incomplete grids fail here.
    """

    column_list = list(columns)
    if len(column_list) != len(set(column_list)):
        raise ValueError("sequence feature columns contain duplicates")

    by_feature: dict[int, dict[int, str]] = {}
    unmatched = []
    for column in column_list:
        match = FEATURE_RE.match(column)
        if match is None:
            unmatched.append(column)
            continue
        feature_id, lag = map(int, match.groups())
        by_feature.setdefault(feature_id, {})[lag] = column
    if unmatched:
        raise ValueError(f"invalid sequence feature names: {sorted(unmatched)}")
    if not by_feature:
        raise ValueError("no sequence features found")

    feature_ids = tuple(sorted(by_feature))
    discovered_lags = tuple(sorted({lag for lag_map in by_feature.values() for lag in lag_map}))
    lag_windows = expected_lags or discovered_lags
    if discovered_lags != tuple(sorted(lag_windows)):
        raise ValueError(
            f"discovered lags {discovered_lags} do not match expected {tuple(sorted(lag_windows))}"
        )
    missing = {
        feature_id: sorted(set(lag_windows) - set(by_feature[feature_id]))
        for feature_id in feature_ids
        if set(by_feature[feature_id]) != set(lag_windows)
    }
    if missing:
        raise ValueError(f"incomplete sequence lag grid: {missing}")

    ordered = tuple(
        by_feature[feature_id][lag] for lag in lag_windows for feature_id in feature_ids
    )
    return SequenceFeatureSchema(
        ordered_columns=ordered,
        feature_ids=feature_ids,
        lag_windows=tuple(lag_windows),
    )


def engineer_lag_differences(
    frame: Any,
    schema: SequenceFeatureSchema,
) -> tuple[Any, tuple[str, ...]]:
    """Add the notebook's four momentum differences per base feature."""

    pl = import_training_dependency("polars")
    expressions = []
    names = []
    by_feature = {
        feature_id: {lag: f"feature_{feature_id}_lag{lag}" for lag in schema.lag_windows}
        for feature_id in schema.feature_ids
    }
    for feature_id in schema.feature_ids:
        for earlier, later in DIFF_PAIRS:
            if earlier not in by_feature[feature_id] or later not in by_feature[feature_id]:
                continue
            name = f"diff_{feature_id}_lag{earlier}_lag{later}"
            expressions.append(
                (
                    pl.col(by_feature[feature_id][earlier]) - pl.col(by_feature[feature_id][later])
                ).alias(name)
            )
            names.append(name)
    if expressions:
        frame = frame.with_columns(expressions)
    return frame, tuple(names)


def engineer_cross_sectional_ranks(
    frame: Any,
    columns: Iterable[str],
    *,
    date_column: str = "date",
) -> tuple[Any, tuple[str, ...]]:
    """Rank engineered values within each point-in-time cross-section."""

    pl = import_training_dependency("polars")
    source_columns = tuple(columns)
    names = tuple(f"csr_{column}" for column in source_columns)
    if not source_columns:
        return frame, names
    if date_column in frame.columns:
        expressions = [
            (
                (pl.col(column).rank("average").over(date_column) - 0.5)
                / pl.len().over(date_column)
            ).alias(name)
            for column, name in zip(source_columns, names, strict=True)
        ]
    else:
        expressions = [
            ((pl.col(column).rank("average") - 0.5) / pl.len()).alias(name)
            for column, name in zip(source_columns, names, strict=True)
        ]
    return frame.with_columns(expressions), names


def sequence_matrix(
    frame: Any,
    schema: SequenceFeatureSchema,
    *,
    neutral_fill: float = 0.5,
) -> np.ndarray:
    """Create the neutral-filled lag-major float32 matrix."""

    return (
        frame.select(schema.ordered_columns)
        .fill_null(neutral_fill)
        .fill_nan(neutral_fill)
        .to_numpy()
        .astype(np.float32)
    )


def validate_prepared_frame(
    frame: Any,
    *,
    key_columns: tuple[str, str] = ("date", "id"),
    target_columns: tuple[str, str] = ("target_10d", "target_30d"),
) -> None:
    """Reject alignment and target failures before any expensive fit."""

    pl = import_training_dependency("polars")
    required = set(key_columns + target_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prepared frame is missing columns: {sorted(missing)}")
    duplicate_count = frame.group_by(list(key_columns)).len().filter(pl.col("len") > 1).height
    if duplicate_count:
        raise ValueError(f"prepared frame has {duplicate_count} duplicate point-in-time keys")
    for target in target_columns:
        values = frame[target].to_numpy()
        if frame[target].null_count() or not np.isfinite(values).all():
            raise ValueError(f"{target} contains null or non-finite values")
        if values.min() < 0.0 or values.max() > 1.0:
            raise ValueError(f"{target} must be ranking labels in [0, 1]")
