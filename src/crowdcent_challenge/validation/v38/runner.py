"""Thin end-to-end v3.8 walk-forward runner."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean

import numpy as np

from crowdcent_challenge.validation.atomic import score_atomic_period
from crowdcent_challenge.validation.eras import HYPERLIQUID_RANKING_ERAS
from crowdcent_challenge.validation.pit import MetaPurpose, latest_available_meta_release

from .book_search import BookSearchConfig, BookSelection, select_scored_book
from .candidates import (
    BlendCandidate,
    CandidateSpec,
    PrimitiveCandidate,
    PriorMetaBlendCandidate,
    ResidualCandidate,
    build_candidate_panel,
    generate_sparse_blends,
)
from .components import COMPONENT_IDS, ComponentFitConfig, fit_component_predictions
from .contracts import MetaInputPanel, PredictionPanel, PreparedDataset
from .dataset import (
    build_meta_panels_for_dataset,
    meta_input_panel_from_store,
    prepared_dataset_from_frame,
)
from .decomposition import BookScoreDecomposition, decompose_book
from .folds import V38NestedFoldConfig, V38OuterFold, build_v38_nested_folds
from .incumbent import V36_SLOT_IDS, build_incumbent_baseline_panels
from .objective import (
    BookPeriodUtility,
    CalibratedSlotPercentile,
    MetricPercentileCalibrator,
    PeriodScore,
    book_period_utility,
    paired_book_delta,
)
from .pool import PoolReferenceConfig, PoolSource, build_pool_reference, score_meta_as_participant


@dataclass(frozen=True)
class FoldRunResult:
    outer_validation_start: date
    outer_validation_end: date
    selected_candidate_ids: tuple[Hashable, ...]
    scored_candidate_ids: tuple[Hashable, ...]
    experimental_candidate_ids: tuple[Hashable, ...]
    inner_selection: BookSelection
    outer_v38_points: float
    outer_v36_points: float
    outer_delta: float
    outer_delta_ci: tuple[float, float]
    inner_periods: int
    outer_periods: int
    inner_decomposition: BookScoreDecomposition
    outer_decomposition: BookScoreDecomposition
    pool_source: str = "own_candidates"
    pool_provenance: str = ""
    pool_models: int = 0
    pool_periods: int = 0


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_index: int
    result: FoldRunResult


@dataclass(frozen=True)
class WalkForwardBacktestResult:
    """Multi-fold historical evaluation under a fixed scoring regime."""

    folds: tuple[WalkForwardFoldResult, ...]
    mean_outer_delta: float
    mean_outer_v38_points: float
    mean_outer_v36_points: float
    counterfactual_use50: bool
    meta_era_start: date
    meta_era_end: date


@dataclass(frozen=True)
class V38RunConfig:
    assets_per_period: int = 40
    n_features: int = 24
    fold_config: V38NestedFoldConfig | None = None
    book_search: BookSearchConfig | None = None
    component_fit: ComponentFitConfig | None = None
    seed: int = 42
    respect_live_era_cutover: bool = False
    pool_reference: "PoolReferenceConfig | None" = None
    fixed_scored_candidate_ids: tuple[Hashable, ...] = ()
    experimental_candidate_ids: tuple[Hashable, ...] = ()


def _resolved_config(config: V38RunConfig | None) -> V38RunConfig:
    base = config or V38RunConfig()
    if base.component_fit is not None:
        return base
    return V38RunConfig(
        assets_per_period=base.assets_per_period,
        n_features=base.n_features,
        fold_config=base.fold_config,
        book_search=base.book_search,
        component_fit=ComponentFitConfig(mode="real", seed=base.seed),
        seed=base.seed,
        respect_live_era_cutover=base.respect_live_era_cutover,
        pool_reference=base.pool_reference,
        fixed_scored_candidate_ids=base.fixed_scored_candidate_ids,
        experimental_candidate_ids=base.experimental_candidate_ids,
    )


def make_synthetic_dataset(config: V38RunConfig | None = None) -> PreparedDataset:
    config = config or V38RunConfig()
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(size=(config.n_features, 2))
    rows: list[tuple[date, str, np.ndarray, np.ndarray]] = []
    start = date(2025, 1, 1)
    for offset in range(700):
        period = start + timedelta(days=offset)
        features = rng.normal(size=(config.assets_per_period, config.n_features))
        noise = rng.normal(scale=0.35, size=(config.assets_per_period, 2))
        scores = features @ weights + noise
        targets = np.empty_like(scores)
        for horizon in range(2):
            order = np.argsort(scores[:, horizon], kind="stable")
            ranks = np.empty(config.assets_per_period, dtype=float)
            ranks[order] = (np.arange(config.assets_per_period) + 0.5) / config.assets_per_period
            targets[:, horizon] = ranks
        for asset_index in range(config.assets_per_period):
            rows.append(
                (
                    period,
                    f"A{asset_index:03d}",
                    targets[asset_index],
                    features[asset_index],
                )
            )
    return PreparedDataset(
        periods=tuple(period for period, _, _, _ in rows),
        ids=tuple(asset_id for _, asset_id, _, _ in rows),
        targets=np.vstack([target for _, _, target, _ in rows]),
        features=np.vstack([feature for _, _, _, feature in rows]),
    )


def run_v38_smoke(config: V38RunConfig | None = None) -> FoldRunResult:
    if config is None:
        config = V38RunConfig(component_fit=ComponentFitConfig(mode="smoke", seed=42))
    elif config.component_fit is None:
        config = V38RunConfig(
            assets_per_period=config.assets_per_period,
            n_features=config.n_features,
            fold_config=config.fold_config,
            book_search=config.book_search,
            component_fit=ComponentFitConfig(mode="smoke", seed=config.seed),
            seed=config.seed,
            respect_live_era_cutover=config.respect_live_era_cutover,
            pool_reference=config.pool_reference,
            fixed_scored_candidate_ids=config.fixed_scored_candidate_ids,
            experimental_candidate_ids=config.experimental_candidate_ids,
        )
    config = _resolved_config(config)
    dataset = make_synthetic_dataset(config)
    specs = candidate_library(smoke=True)
    folds = build_v38_nested_folds(
        dataset.unique_periods,
        config.fold_config
        or V38NestedFoldConfig(
            outer_min_train_days=360,
            outer_validation_days=60,
            outer_step_days=60,
            outer_max_folds=1,
            inner_min_train_days=180,
            inner_validation_days=45,
            inner_step_days=45,
            inner_folds=1,
            embargo_days=30,  # frozen synthetic benchmark; live default is 31
        ),
    )
    if not folds:
        raise RuntimeError("no nested folds were built for the synthetic dataset")
    return run_v38_outer_fold(dataset, folds[0], config=config, specs=specs)


def run_v38_from_dataset(
    dataset: PreparedDataset,
    meta_frame=None,
    *,
    config: V38RunConfig | None = None,
    specs: Sequence[CandidateSpec] | None = None,
) -> FoldRunResult:
    """Run the first nested outer fold on a prepared real-data panel."""

    config = _resolved_config(config)
    specs = list(specs or candidate_library())
    folds = build_v38_nested_folds(
        dataset.unique_periods,
        config.fold_config
        or V38NestedFoldConfig(
            outer_min_train_days=360,
            outer_validation_days=60,
            outer_step_days=60,
            outer_max_folds=1,
            inner_min_train_days=180,
            inner_validation_days=45,
            inner_step_days=45,
            inner_folds=1,
            embargo_days=31,
        ),
    )
    if not folds:
        raise RuntimeError("no nested folds were built for the dataset")
    return run_v38_outer_fold(
        dataset,
        folds[0],
        config=config,
        specs=specs,
        meta_frame=meta_frame,
    )


def compact_walkforward_fold_config(*, max_folds: int = 3) -> V38NestedFoldConfig:
    """Compact folds for the ~210-day real meta-era panel."""

    return V38NestedFoldConfig(
        outer_min_train_days=110,
        outer_validation_days=25,
        outer_step_days=25,
        outer_max_folds=max_folds,
        inner_min_train_days=50,
        inner_validation_days=20,
        inner_step_days=20,
        inner_folds=2,
        embargo_days=31,
    )


def _auto_fold_config(dataset: PreparedDataset, *, max_folds: int = 3) -> V38NestedFoldConfig:
    panel = meta_era_dataset(dataset)
    span = (panel.unique_periods[-1] - panel.unique_periods[0]).days
    if span < 400:
        return compact_walkforward_fold_config(max_folds=max_folds)
    return default_walkforward_fold_config(max_folds=max_folds)


def default_walkforward_fold_config(*, max_folds: int = 6) -> V38NestedFoldConfig:
    """Conservative walk-forward defaults for meta-era historical backtests."""

    return V38NestedFoldConfig(
        outer_min_train_days=360,
        outer_validation_days=90,
        outer_step_days=90,
        outer_max_folds=max_folds,
        inner_min_train_days=180,
        inner_validation_days=45,
        inner_step_days=45,
        inner_folds=2,
        embargo_days=31,
    )


def meta_era_dataset(dataset: PreparedDataset) -> PreparedDataset:
    """Restrict a panel to periods with point-in-time meta-model coverage."""

    mask = dataset.mask_between(
        HYPERLIQUID_RANKING_ERAS.meta_model_start,
        dataset.unique_periods[-1],
    )
    return dataset.take(mask)


def run_v38_walkforward_backtest(
    dataset: PreparedDataset,
    meta_frame=None,
    *,
    config: V38RunConfig | None = None,
    specs: Sequence[CandidateSpec] | None = None,
    max_folds: int = 6,
    fold_indices: Sequence[int] | None = None,
) -> WalkForwardBacktestResult:
    """Walk-forward v3.8 vs v3.6 under counterfactual USE50 by default."""

    import logging

    config = _resolved_config(config)
    specs = list(specs or candidate_library())
    panel = meta_era_dataset(dataset)
    folds = build_v38_nested_folds(
        panel.unique_periods,
        config.fold_config or _auto_fold_config(dataset, max_folds=max_folds),
    )
    if not folds:
        raise RuntimeError("no walk-forward folds fit the meta-era panel")
    selected_indices = tuple(range(len(folds))) if fold_indices is None else tuple(fold_indices)
    invalid_indices = [index for index in selected_indices if index < 0 or index >= len(folds)]
    if invalid_indices:
        raise ValueError(
            f"fold indices {invalid_indices} outside available range 0..{len(folds) - 1}"
        )
    if not selected_indices:
        raise ValueError("at least one fold index is required")

    logger = logging.getLogger(__name__)
    results: list[WalkForwardFoldResult] = []
    for index in selected_indices:
        nested = folds[index]
        logger.info("walk-forward fold %d/%d started", index + 1, len(folds))
        result = run_v38_outer_fold(
            panel,
            nested,
            config=config,
            specs=specs,
            meta_frame=meta_frame,
        )
        results.append(WalkForwardFoldResult(fold_index=index, result=result))
        logger.info(
            "walk-forward fold %d/%d complete: v3.8=%+.4f v3.6=%+.4f delta=%+.4f",
            index + 1,
            len(folds),
            result.outer_v38_points,
            result.outer_v36_points,
            result.outer_delta,
        )
    result_tuple = tuple(results)
    return WalkForwardBacktestResult(
        folds=result_tuple,
        mean_outer_delta=fmean(fold.result.outer_delta for fold in result_tuple),
        mean_outer_v38_points=fmean(fold.result.outer_v38_points for fold in result_tuple),
        mean_outer_v36_points=fmean(fold.result.outer_v36_points for fold in result_tuple),
        counterfactual_use50=not config.respect_live_era_cutover,
        meta_era_start=panel.unique_periods[0],
        meta_era_end=panel.unique_periods[-1],
    )


def run_v38_outer_fold(
    dataset: PreparedDataset,
    nested: V38OuterFold,
    *,
    config: V38RunConfig | None = None,
    specs: Sequence[CandidateSpec] | None = None,
    meta_frame=None,
) -> FoldRunResult:
    config = _resolved_config(config)
    specs = list(specs or candidate_library())
    inner_atomic: dict[Hashable, list[PeriodScore]] = {}

    for inner in nested.inner:
        inner_train = dataset.take(dataset.mask_between(inner.train_start, inner.train_end))
        inner_val = dataset.take(dataset.mask_between(inner.validation_start, inner.validation_end))
        fold_scores = _evaluate_split(
            inner_train,
            inner_val,
            specs,
            config=config,
            meta_frame=meta_frame,
        )
        for candidate_id, period_scores in fold_scores.items():
            inner_atomic.setdefault(candidate_id, []).extend(period_scores)

    reference_periods = {
        period_score.period for period_scores in inner_atomic.values() for period_score in period_scores
    }
    if config.pool_reference is None:
        reference_scores = [
            period_score.score
            for period_scores in inner_atomic.values()
            for period_score in period_scores
        ]
        calibrator = MetricPercentileCalibrator.fit(
            reference_scores,
            respect_live_era_cutover=config.respect_live_era_cutover,
        )
        pool_source = PoolSource.OWN_CANDIDATES.value
        pool_provenance = (
            f"legacy inner-oos: {len(inner_atomic)} candidates over "
            f"{len(reference_periods)} periods ({len(reference_scores)} scores)"
        )
        pool_models = len(inner_atomic)
        pool_periods = len(reference_periods)
    else:
        pool_cfg = config.pool_reference
        meta_participant = None
        if (
            pool_cfg.source is PoolSource.META_ANCHORED
            and pool_cfg.include_meta_participant
            and meta_frame is not None
            and reference_periods
        ):
            inner_ref_panel = dataset.take(
                np.asarray([period in reference_periods for period in dataset.periods])
            )
            meta_participant = score_meta_as_participant(
                inner_ref_panel,
                meta_frame,
                min_assets=pool_cfg.min_assets,
            )
        pool = build_pool_reference(
            config=pool_cfg,
            inner_atomic=inner_atomic,
            incumbent_ids=V36_SLOT_IDS,
            meta_participant_scores=meta_participant,
            reference_periods=reference_periods,
        )
        calibrator = MetricPercentileCalibrator.fit(
            pool.scores,
            respect_live_era_cutover=config.respect_live_era_cutover,
        )
        pool_source = pool.source.value
        pool_provenance = pool.provenance
        pool_models = pool.n_models
        pool_periods = pool.n_periods
    inner_rows = {
        candidate_id: [
            calibrator.calibrate(
                candidate_id=candidate_id,
                period=period_score.period,
                score=period_score.score,
            )
            for period_score in period_scores
        ]
        for candidate_id, period_scores in inner_atomic.items()
    }
    search_rows = {
        candidate_id: rows
        for candidate_id, rows in inner_rows.items()
        if candidate_id not in V36_SLOT_IDS
    }
    search_config = config.book_search or BookSearchConfig(bootstrap_samples=200)
    inner_selection = select_scored_book(
        inner_rows,
        incumbent_ids=V36_SLOT_IDS,
        search_ids=tuple(search_rows),
        fixed_scored_ids=config.fixed_scored_candidate_ids,
        experimental_ids=config.experimental_candidate_ids,
        config=search_config,
    )

    outer_train = dataset.take(
        dataset.mask_between(nested.outer.train_start, nested.outer.train_end)
    )
    outer_val = dataset.take(
        dataset.mask_between(nested.outer.validation_start, nested.outer.validation_end)
    )
    outer_scores = _evaluate_split(
        outer_train,
        outer_val,
        specs,
        config=config,
        meta_frame=meta_frame,
    )
    outer_rows = {
        candidate_id: [
            calibrator.calibrate(
                candidate_id=candidate_id,
                period=period_score.period,
                score=period_score.score,
            )
            for period_score in period_scores
        ]
        for candidate_id, period_scores in outer_scores.items()
    }

    v38_utilities = book_utilities(
        outer_rows,
        inner_selection.selected.candidate_ids,
    )
    v36_utilities = book_utilities(outer_rows, V36_SLOT_IDS)
    delta = paired_book_delta(
        v38_utilities,
        v36_utilities,
        block_length=search_config.block_length,
        samples=search_config.bootstrap_samples,
        confidence=search_config.confidence,
        seed=config.seed,
    )
    scored_ids = inner_selection.selected.candidate_ids
    experimental_ids = inner_selection.experimental_ids
    inner_decomposition = decompose_book(
        calibrator,
        inner_atomic,
        scored_ids=scored_ids,
        experimental_ids=experimental_ids,
        scope="inner",
    )
    outer_decomposition = decompose_book(
        calibrator,
        outer_scores,
        scored_ids=scored_ids,
        experimental_ids=experimental_ids,
        scope="outer",
    )
    return FoldRunResult(
        outer_validation_start=nested.outer.validation_start,
        outer_validation_end=nested.outer.validation_end,
        selected_candidate_ids=scored_ids,
        scored_candidate_ids=scored_ids,
        experimental_candidate_ids=experimental_ids,
        inner_selection=inner_selection,
        outer_v38_points=fmean(row.points for row in v38_utilities),
        outer_v36_points=fmean(row.points for row in v36_utilities),
        outer_delta=delta.estimate,
        outer_delta_ci=(delta.lower, delta.upper),
        inner_periods=len(next(iter(inner_rows.values()))),
        outer_periods=len(v38_utilities),
        inner_decomposition=inner_decomposition,
        outer_decomposition=outer_decomposition,
        pool_source=pool_source,
        pool_provenance=pool_provenance,
        pool_models=pool_models,
        pool_periods=pool_periods,
    )


def candidate_library(*, smoke: bool = False) -> list[CandidateSpec]:
    if smoke:
        specs: list[CandidateSpec] = [
            PrimitiveCandidate(component_id) for component_id in COMPONENT_IDS
        ]
        specs.append(
            BlendCandidate(
                component_ids=("xgb", "transformer"),
                weights_10d=(0.50, 0.50),
                weights_30d=(0.50, 0.50),
            )
        )
        specs.extend(
            [
                ResidualCandidate(PrimitiveCandidate("xgb"), 0.50, 0.50),
                ResidualCandidate(PrimitiveCandidate("transformer"), 0.50, 0.50),
            ]
        )
        return specs

    specs = list(generate_sparse_blends(COMPONENT_IDS))
    specs.extend(
        [
            PriorMetaBlendCandidate(PrimitiveCandidate("xgb"), 0.25, 0.25),
            PriorMetaBlendCandidate(PrimitiveCandidate("transformer"), 0.25, 0.25),
            ResidualCandidate(PrimitiveCandidate("xgb"), 0.50, 0.50),
            ResidualCandidate(PrimitiveCandidate("transformer"), 0.50, 0.50),
        ]
    )
    return specs


def book_utilities(
    rows: Mapping[Hashable, Sequence[CalibratedSlotPercentile]],
    candidate_ids: tuple[Hashable, ...],
) -> tuple[BookPeriodUtility, ...]:
    by_period: dict[date, list[CalibratedSlotPercentile]] = {}
    for candidate_id in candidate_ids:
        for row in rows[candidate_id]:
            by_period.setdefault(row.period, []).append(row)
    return tuple(
        book_period_utility(by_period[period])
        for period in sorted(by_period)
        if len(by_period[period]) == len(candidate_ids)
    )


def _evaluate_split(
    train: PreparedDataset,
    validation: PreparedDataset,
    specs: Sequence[CandidateSpec],
    *,
    config: V38RunConfig,
    meta_frame=None,
) -> dict[Hashable, list[PeriodScore]]:
    component_bundle = fit_component_predictions(
        train,
        validation,
        config=config.component_fit or ComponentFitConfig(mode="real", seed=config.seed),
    )
    components = component_bundle.panels
    if meta_frame is None:
        score_meta, input_meta = build_meta_history(components, validation.unique_periods)
        meta_input = meta_input_panel(components["xgb"], input_meta)
    else:
        score_meta, input_meta = build_meta_panels_for_dataset(validation, meta_frame)
        meta_input = meta_input_panel_from_store(components["xgb"], input_meta)
    scores: dict[Hashable, list[PeriodScore]] = {}

    for spec in specs:
        panel = build_candidate_panel(spec, components, meta_input=meta_input)
        scores[spec.candidate_id] = score_panel(validation, panel, score_meta)

    for slot_id, panel in build_incumbent_baseline_panels(components, meta_input).items():
        scores[slot_id] = score_panel(validation, panel, score_meta)
    return scores


def build_meta_history(
    components: Mapping[str, PredictionPanel],
    periods: Sequence[date],
) -> tuple[dict[date, PredictionPanel], dict[date, PredictionPanel]]:
    xgb = components["xgb"]
    transformer = components["transformer"]
    score_meta: dict[date, PredictionPanel] = {}
    input_meta: dict[date, PredictionPanel] = {}
    for period in periods:
        mask = np.asarray([row_period == period for row_period in xgb.periods])
        ids = tuple(asset_id for asset_id, keep in zip(xgb.ids, mask, strict=True) if keep)
        blended = 0.5 * xgb.values[mask] + 0.5 * transformer.values[mask]
        score_meta[period] = PredictionPanel(
            periods=tuple(period for _ in ids),
            ids=ids,
            values=blended,
        )
        prior = latest_available_meta_release(
            tuple(score_meta),
            period,
            MetaPurpose.MODEL_INPUT,
        )
        if prior is None:
            values = np.full_like(blended, 0.5)
        else:
            prior_lookup = dict(zip(score_meta[prior].ids, score_meta[prior].values, strict=True))
            values = np.vstack([prior_lookup[asset_id] for asset_id in ids])
        input_meta[period] = PredictionPanel(
            periods=tuple(period for _ in ids),
            ids=ids,
            values=values,
        )
    return score_meta, input_meta


def meta_input_panel(
    reference: PredictionPanel,
    input_meta: Mapping[date, PredictionPanel],
) -> MetaInputPanel:
    release_dates: list[date] = []
    values: list[np.ndarray] = []
    for period, asset_id in zip(reference.periods, reference.ids, strict=True):
        meta_panel = input_meta[period]
        lookup = dict(zip(meta_panel.ids, meta_panel.values, strict=True))
        release = latest_available_meta_release(
            input_meta,
            period,
            MetaPurpose.MODEL_INPUT,
        )
        release_dates.append(release or period - timedelta(days=1))
        values.append(lookup[asset_id])
    return MetaInputPanel(
        predictions=PredictionPanel(
            periods=reference.periods,
            ids=reference.ids,
            values=np.vstack(values),
        ),
        release_dates=tuple(release_dates),
    )


def score_panel(
    dataset: PreparedDataset,
    panel: PredictionPanel,
    score_meta: Mapping[date, PredictionPanel],
) -> list[PeriodScore]:
    lookup = dict(zip(panel.keys, panel.values, strict=True))
    rows: list[PeriodScore] = []
    for period in sorted(set(dataset.periods)):
        if period < HYPERLIQUID_RANKING_ERAS.meta_model_start:
            continue
        if period not in score_meta:
            continue
        mask = np.asarray([row_period == period for row_period in dataset.periods])
        ids = [dataset.ids[index] for index in np.flatnonzero(mask)]
        if len(ids) < 10:
            continue
        pred_10d = np.asarray([lookup[(period, asset_id)][0] for asset_id in ids], dtype=float)
        pred_30d = np.asarray([lookup[(period, asset_id)][1] for asset_id in ids], dtype=float)
        meta_panel = score_meta[period]
        meta_lookup = dict(zip(meta_panel.ids, meta_panel.values, strict=True))
        score = score_atomic_period(
            y_true_10d=dataset.targets[mask, 0],
            y_pred_10d=pred_10d,
            y_true_30d=dataset.targets[mask, 1],
            y_pred_30d=pred_30d,
            meta_pred_10d=np.asarray([meta_lookup[asset_id][0] for asset_id in ids], dtype=float),
            meta_pred_30d=np.asarray([meta_lookup[asset_id][1] for asset_id in ids], dtype=float),
        )
        rows.append(PeriodScore(period=period, score=score))
    return rows


def decomposition_summary(result: WalkForwardBacktestResult) -> dict:
    """Serialize walk-forward output for JSON logging."""

    def fold_payload(fold: WalkForwardFoldResult) -> dict:
        outer = fold.result.outer_decomposition
        return {
            "fold_index": fold.fold_index,
            "outer_validation_start": fold.result.outer_validation_start.isoformat(),
            "outer_validation_end": fold.result.outer_validation_end.isoformat(),
            "selected_candidate_ids": [str(value) for value in fold.result.selected_candidate_ids],
            "outer_v38_points": fold.result.outer_v38_points,
            "outer_v36_points": fold.result.outer_v36_points,
            "outer_delta": fold.result.outer_delta,
            "outer_delta_ci": fold.result.outer_delta_ci,
            "pool_source": fold.result.pool_source,
            "pool_provenance": fold.result.pool_provenance,
            "pool_models": fold.result.pool_models,
            "pool_periods": fold.result.pool_periods,
            "book_points": outer.book_points,
            "book_points_percentile": outer.book_points_percentile,
            "raw_unique_split": {
                "raw_percentile": outer.raw_unique_split.raw_percentile,
                "unique_percentile": outer.raw_unique_split.unique_percentile,
                "raw_contribution": outer.raw_unique_split.raw_contribution,
                "unique_contribution": outer.raw_unique_split.unique_contribution,
            },
            "metric_contributions": [
                {
                    "metric": row.metric,
                    "side": row.side,
                    "mean_percentile": row.mean_percentile,
                    "percentile_contribution": row.percentile_contribution,
                }
                for row in outer.metric_contributions
            ],
            "slot_contributions": [
                {
                    "candidate_id": str(row.candidate_id),
                    "role": row.role,
                    "mean_points_percentile": row.mean_points_percentile,
                    "raw_percentile": row.raw_percentile,
                    "unique_percentile": row.unique_percentile,
                    "share_of_book": row.share_of_book,
                    "loo_book_points": row.loo_book_points,
                    "loo_points_delta": row.loo_points_delta,
                    "promotion": (
                        {
                            "promote": row.promotion.promote,
                            "matched_periods": row.promotion.matched_periods,
                            "candidate_mean": row.promotion.candidate_mean,
                            "incumbent_mean": row.promotion.incumbent_mean,
                            "underperformance_frequency": (
                                row.promotion.underperformance_frequency
                            ),
                            "reasons": list(row.promotion.reasons),
                        }
                        if row.promotion is not None
                        else None
                    ),
                }
                for row in outer.slot_contributions
            ],
        }

    return {
        "counterfactual_use50": result.counterfactual_use50,
        "meta_era_start": result.meta_era_start.isoformat(),
        "meta_era_end": result.meta_era_end.isoformat(),
        "mean_outer_delta": result.mean_outer_delta,
        "mean_outer_v38_points": result.mean_outer_v38_points,
        "mean_outer_v36_points": result.mean_outer_v36_points,
        "folds": [fold_payload(fold) for fold in result.folds],
    }


def run_v38_pool_calibrated_backtest(
    *,
    train_path: str | Path = "/var/lib/centaur/workspace/.tmp/v38_cc_train.parquet",
    meta_path: str | Path | None = None,
    pool_source: PoolSource = PoolSource.META_ANCHORED,
    component_mode: str = "real",
    fold_config: V38NestedFoldConfig | None = None,
    max_folds: int = 3,
    fold_index: int | None = None,
    fixed_scored_candidate_ids: tuple[Hashable, ...] = (),
    max_experimental_slots: int = 0,
    experimental_candidate_ids: tuple[Hashable, ...] = (),
    confidence: float = 0.90,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> WalkForwardBacktestResult:
    """Run pool-calibrated walk-forward on real Hyperliquid data.

    ``fold_index`` runs one fold in isolation so callers can checkpoint each
    fold in a fresh process and resume long real-model backtests safely.
    """

    import json
    import logging
    import os
    from pathlib import Path as PathType

    from .dataset import load_meta_frame

    logger = logging.getLogger(__name__)
    if os.environ.get("KERAS_BACKEND") != "jax":
        logger.warning(
            "KERAS_BACKEND is not jax; sequence models may proxy to XGB unless set before import"
        )

    pl = __import__("polars")
    meta_frame = load_meta_frame(
        preferred_paths=[PathType(meta_path)] if meta_path is not None else None,
    )
    dataset = prepared_dataset_from_frame(pl.read_parquet(train_path))
    config = V38RunConfig(
        component_fit=ComponentFitConfig(mode=component_mode, seed=seed),
        book_search=BookSearchConfig(
            min_scored_slots=1,
            max_scored_slots=5,
            max_experimental_slots=max_experimental_slots,
            bootstrap_samples=2000,
            confidence=confidence,
            seed=seed,
        ),
        pool_reference=PoolReferenceConfig(source=pool_source),
        fold_config=fold_config or _auto_fold_config(dataset, max_folds=max_folds),
        seed=seed,
        fixed_scored_candidate_ids=fixed_scored_candidate_ids,
        experimental_candidate_ids=experimental_candidate_ids,
    )
    result = run_v38_walkforward_backtest(
        dataset,
        meta_frame,
        config=config,
        max_folds=max_folds,
        fold_indices=None if fold_index is None else (fold_index,),
    )
    if output_path is not None:
        destination = PathType(output_path)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(json.dumps(decomposition_summary(result), indent=2))
        temporary.replace(destination)
    return result
