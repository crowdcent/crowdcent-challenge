"""Tournament-pool reference distributions for live-comparable calibration."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np

from crowdcent_challenge.validation.atomic import (
    RAW_METRICS,
    UNIQUE_METRICS,
    AtomicPeriodScore,
    score_atomic_period,
)
from crowdcent_challenge.validation.pit import MetaPurpose, latest_available_meta_release

from .contracts import PreparedDataset
from .dataset import build_meta_panels_for_dataset
from .objective import PeriodScore

CACHED_POOL_COLUMNS = ("model_id", "period", *RAW_METRICS, *UNIQUE_METRICS)


class PoolSource(str, Enum):
    CACHED = "cached"
    META_ANCHORED = "meta_anchored"
    OWN_CANDIDATES = "own_candidates"


@dataclass(frozen=True)
class PoolReferenceConfig:
    source: PoolSource = PoolSource.OWN_CANDIDATES
    include_own_candidates: bool = True
    include_incumbent_slots: bool = False
    include_meta_participant: bool = True
    cached_pool_path: Path | None = None
    min_reference_models: int = 1
    min_reference_scores: int = 30
    min_assets: int = 10
    fallback: tuple[PoolSource, ...] = (PoolSource.OWN_CANDIDATES,)


@dataclass(frozen=True)
class PoolReference:
    scores: tuple[AtomicPeriodScore, ...]
    source: PoolSource
    n_models: int
    n_periods: int
    n_scores: int
    provenance: str


def _import_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("pool cached loading requires polars") from exc
    return pl


def score_meta_as_participant(
    dataset: PreparedDataset,
    meta_frame,
    *,
    min_assets: int = 10,
) -> list[AtomicPeriodScore]:
    """Score the PIT meta-model as one pool participant."""

    pl = _import_polars()
    meta = meta_frame.with_columns(pl.col("release_date").cast(pl.Date))
    release_dates = tuple(sorted(meta["release_date"].unique().to_list()))
    score_meta, input_meta = build_meta_panels_for_dataset(dataset, meta_frame)
    rows: list[AtomicPeriodScore] = []
    for period in dataset.unique_periods:
        if period not in score_meta or period not in input_meta:
            continue
        if latest_available_meta_release(release_dates, period, MetaPurpose.MODEL_INPUT) is None:
            continue
        mask = np.asarray([row_period == period for row_period in dataset.periods])
        ids = [dataset.ids[index] for index in np.flatnonzero(mask)]
        if len(ids) < min_assets:
            continue
        meta_panel = score_meta[period]
        uniqueness_panel = input_meta[period]
        meta_lookup = dict(zip(meta_panel.ids, meta_panel.values, strict=True))
        unique_lookup = dict(zip(uniqueness_panel.ids, uniqueness_panel.values, strict=True))
        pred_10d = np.asarray([meta_lookup[asset_id][0] for asset_id in ids], dtype=float)
        pred_30d = np.asarray([meta_lookup[asset_id][1] for asset_id in ids], dtype=float)
        score = score_atomic_period(
            y_true_10d=dataset.targets[mask, 0],
            y_pred_10d=pred_10d,
            y_true_30d=dataset.targets[mask, 1],
            y_pred_30d=pred_30d,
            meta_pred_10d=np.asarray([unique_lookup[asset_id][0] for asset_id in ids], dtype=float),
            meta_pred_30d=np.asarray([unique_lookup[asset_id][1] for asset_id in ids], dtype=float),
            min_assets=min_assets,
        )
        if score.unique is None or not _score_is_finite(score):
            continue
        rows.append(score)
    return rows


def load_cached_pool_scores(
    path: Path,
    *,
    reference_periods: Collection[date],
) -> list[AtomicPeriodScore]:
    """Load a per-(model, period) atomic-metric table."""

    pl = _import_polars()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).with_columns(pl.col("period").cast(pl.Date))
    missing = set(CACHED_POOL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"cached pool missing columns: {sorted(missing)}")
    allowed = set(reference_periods)
    frame = frame.filter(pl.col("period").is_in(list(allowed)))
    rows: list[AtomicPeriodScore] = []
    for row in frame.iter_rows(named=True):
        raw = {metric: float(row[metric]) for metric in RAW_METRICS}
        unique = {metric: float(row[metric]) for metric in UNIQUE_METRICS}
        score = AtomicPeriodScore(raw=raw, unique=unique)
        if _score_is_finite(score):
            rows.append(score)
    return rows


def _own_candidate_scores(
    inner_atomic: Mapping[Hashable, Sequence[PeriodScore]],
    *,
    incumbent_ids: tuple[Hashable, ...],
    include_incumbent_slots: bool,
    reference_periods: Collection[date] | None,
) -> tuple[list[AtomicPeriodScore], set[Hashable], set[date]]:
    allowed = set(reference_periods) if reference_periods is not None else None
    scores: list[AtomicPeriodScore] = []
    models: set[Hashable] = set()
    periods: set[date] = set()
    for candidate_id, period_scores in inner_atomic.items():
        if not include_incumbent_slots and candidate_id in incumbent_ids:
            continue
        models.add(candidate_id)
        for period_score in period_scores:
            if allowed is not None and period_score.period not in allowed:
                continue
            if period_score.score.unique is None:
                continue
            scores.append(period_score.score)
            periods.add(period_score.period)
    return scores, models, periods


def _score_is_finite(score: AtomicPeriodScore) -> bool:
    if not all(np.isfinite(value) for value in score.raw.values()):
        return False
    if score.unique is None:
        return False
    return all(np.isfinite(value) for value in score.unique.values())


def _pool_from_source(
    source: PoolSource,
    *,
    config: PoolReferenceConfig,
    inner_atomic: Mapping[Hashable, Sequence[PeriodScore]],
    incumbent_ids: tuple[Hashable, ...],
    meta_participant_scores: Sequence[AtomicPeriodScore] | None,
    reference_periods: Collection[date] | None,
) -> PoolReference | None:
    if source is PoolSource.CACHED:
        if config.cached_pool_path is None:
            return None
        try:
            scores = load_cached_pool_scores(
                config.cached_pool_path,
                reference_periods=reference_periods or (),
            )
        except FileNotFoundError:
            return None
        pl = _import_polars()
        frame = pl.read_parquet(config.cached_pool_path).with_columns(
            pl.col("period").cast(pl.Date)
        )
        if reference_periods is not None:
            frame = frame.filter(pl.col("period").is_in(list(reference_periods)))
        n_models = frame.select("model_id").n_unique()
        n_periods = frame.select("period").n_unique()
        return PoolReference(
            scores=tuple(scores),
            source=source,
            n_models=n_models,
            n_periods=n_periods,
            n_scores=len(scores),
            provenance=(
                f"cached: {n_models} models over {n_periods} periods "
                f"({len(scores)} scores)"
            ),
        )

    own_scores, own_models, own_periods = _own_candidate_scores(
        inner_atomic,
        incumbent_ids=incumbent_ids,
        include_incumbent_slots=config.include_incumbent_slots,
        reference_periods=reference_periods,
    )
    if source is PoolSource.OWN_CANDIDATES:
        if not config.include_own_candidates or not own_scores:
            return None
        return PoolReference(
            scores=tuple(own_scores),
            source=source,
            n_models=len(own_models),
            n_periods=len(own_periods),
            n_scores=len(own_scores),
            provenance=(
                f"own_candidates: {len(own_models)} models over "
                f"{len(own_periods)} periods ({len(own_scores)} scores)"
            ),
        )

    if source is PoolSource.META_ANCHORED:
        scores = list(own_scores) if config.include_own_candidates else []
        n_models = len(own_models) if config.include_own_candidates else 0
        meta_scores = list(meta_participant_scores or ())
        if config.include_meta_participant:
            if not meta_scores:
                return None
            scores.extend(meta_scores)
            n_models += 1
        if not scores:
            return None
        return PoolReference(
            scores=tuple(scores),
            source=source,
            n_models=n_models,
            n_periods=len(own_periods) if own_periods else len(meta_scores),
            n_scores=len(scores),
            provenance=(
                f"meta_anchored: {len(own_models) if config.include_own_candidates else 0} candidate models"
                f"{' + meta' if meta_scores else ''} over "
                f"{len(own_periods) if own_periods else len(meta_scores)} periods "
                f"({len(scores)} scores)"
            ),
        )

    return None


def build_pool_reference(
    *,
    config: PoolReferenceConfig,
    inner_atomic: Mapping[Hashable, Sequence[PeriodScore]],
    incumbent_ids: tuple[Hashable, ...] = (),
    meta_participant_scores: Sequence[AtomicPeriodScore] | None = None,
    reference_periods: Collection[date] | None = None,
) -> PoolReference:
    """Assemble the fixed reference population for one outer fold."""

    chain = (config.source, *config.fallback)
    seen: set[PoolSource] = set()
    last_error = "no pool source produced reference scores"
    for source in chain:
        if source in seen:
            continue
        seen.add(source)
        pool = _pool_from_source(
            source,
            config=config,
            inner_atomic=inner_atomic,
            incumbent_ids=incumbent_ids,
            meta_participant_scores=meta_participant_scores,
            reference_periods=reference_periods,
        )
        if pool is None:
            continue
        if (
            pool.n_models < config.min_reference_models
            or pool.n_scores < config.min_reference_scores
        ):
            last_error = (
                f"{source.value} too thin ({pool.n_models} models, {pool.n_scores} scores)"
            )
            continue
        if source is not config.source:
            return PoolReference(
                scores=pool.scores,
                source=pool.source,
                n_models=pool.n_models,
                n_periods=pool.n_periods,
                n_scores=pool.n_scores,
                provenance=f"fallback from {config.source.value}: {pool.provenance}",
            )
        return pool

    own = _pool_from_source(
        PoolSource.OWN_CANDIDATES,
        config=config,
        inner_atomic=inner_atomic,
        incumbent_ids=incumbent_ids,
        meta_participant_scores=None,
        reference_periods=reference_periods,
    )
    if own is None or not own.scores:
        raise ValueError(last_error)
    return PoolReference(
        scores=own.scores,
        source=PoolSource.OWN_CANDIDATES,
        n_models=own.n_models,
        n_periods=own.n_periods,
        n_scores=own.n_scores,
        provenance=f"fallback from {config.source.value}: {own.provenance}",
    )
