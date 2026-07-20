"""Exhaustive scored-book selection on paired inner-OOS utility."""

import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .objective import (
    BookPeriodUtility,
    BootstrapMean,
    CalibratedSlotPercentile,
    book_period_utility,
    cosine_points,
    moving_block_bootstrap_weights,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookSearchConfig:
    max_scored_slots: int = 5
    min_scored_slots: int = 1
    max_experimental_slots: int = 0
    block_length: int = 30
    bootstrap_samples: int = 2000
    confidence: float = 0.80
    seed: int = 42

    def __post_init__(self) -> None:
        if not 1 <= self.max_scored_slots <= 5:
            raise ValueError("max_scored_slots must be in [1, 5]")
        if not 1 <= self.min_scored_slots <= self.max_scored_slots:
            raise ValueError("min_scored_slots must be in [1, max_scored_slots]")
        if self.max_experimental_slots < 0:
            raise ValueError("max_experimental_slots must be >= 0")


@dataclass(frozen=True)
class BookEvaluation:
    candidate_ids: tuple[Hashable, ...]
    period_utilities: tuple[BookPeriodUtility, ...]
    delta_vs_incumbent: BootstrapMean


@dataclass(frozen=True)
class BookSelection:
    incumbent_ids: tuple[Hashable, ...]
    selected: BookEvaluation
    experimental_ids: tuple[Hashable, ...]
    evaluations: tuple[BookEvaluation, ...]


def _index_candidate_rows(
    candidate_rows: Mapping[Hashable, Sequence[CalibratedSlotPercentile]],
) -> tuple[dict[Hashable, dict[object, CalibratedSlotPercentile]], tuple[object, ...]]:
    if not candidate_rows:
        raise ValueError("candidate_rows cannot be empty")
    indexed = {}
    common_periods = None
    for candidate_id, rows in candidate_rows.items():
        by_period = {}
        for row in rows:
            if row.candidate_id != candidate_id:
                raise ValueError("candidate row is filed under the wrong candidate id")
            if row.period in by_period:
                raise ValueError("candidate has duplicate period rows")
            by_period[row.period] = row
        if not by_period:
            raise ValueError("every candidate needs at least one period")
        periods = set(by_period)
        common_periods = periods if common_periods is None else common_periods & periods
        indexed[candidate_id] = by_period
    if not common_periods:
        raise ValueError("candidates have no common scored periods")
    return indexed, tuple(sorted(common_periods))


def select_scored_book(
    candidate_rows: Mapping[Hashable, Sequence[CalibratedSlotPercentile]],
    *,
    incumbent_ids: tuple[Hashable, ...],
    search_ids: tuple[Hashable, ...] | None = None,
    fixed_scored_ids: tuple[Hashable, ...] | None = None,
    experimental_ids: tuple[Hashable, ...] | None = None,
    config: BookSearchConfig | None = None,
) -> BookSelection:
    """Enumerate books and maximize the lower bound of paired points delta."""

    config = config or BookSearchConfig()
    indexed, common_periods = _index_candidate_rows(candidate_rows)
    if not incumbent_ids or len(set(incumbent_ids)) != len(incumbent_ids):
        raise ValueError("incumbent_ids must be non-empty and unique")
    missing_incumbents = set(incumbent_ids) - set(indexed)
    if missing_incumbents:
        raise ValueError(f"missing incumbent candidates: {sorted(missing_incumbents)}")
    search_pool = tuple(search_ids) if search_ids is not None else tuple(indexed)
    missing_search = set(search_pool) - set(indexed)
    if missing_search:
        raise ValueError(f"missing search candidates: {sorted(missing_search)}")
    fixed_book = tuple(fixed_scored_ids or ())
    if fixed_book:
        if len(set(fixed_book)) != len(fixed_book):
            raise ValueError("fixed_scored_ids must be unique")
        missing_fixed = set(fixed_book) - set(search_pool)
        if missing_fixed:
            raise ValueError(f"fixed scored candidates are not searchable: {sorted(missing_fixed)}")
        if not config.min_scored_slots <= len(fixed_book) <= config.max_scored_slots:
            raise ValueError("fixed scored book size is outside configured slot bounds")
        fixed_set = set(fixed_book)
        fixed_book = tuple(candidate_id for candidate_id in search_pool if candidate_id in fixed_set)
    nominees = tuple(experimental_ids or ())
    if len(set(nominees)) != len(nominees):
        raise ValueError("experimental_ids must be unique")
    missing_nominees = set(nominees) - set(indexed)
    if missing_nominees:
        raise ValueError(f"missing experimental candidates: {sorted(missing_nominees)}")

    def utilities(candidate_ids: tuple[Hashable, ...]) -> tuple[BookPeriodUtility, ...]:
        return tuple(
            book_period_utility([indexed[candidate_id][period] for candidate_id in candidate_ids])
            for period in common_periods
        )

    incumbent_utilities = utilities(incumbent_ids)
    incumbent_points = np.asarray([row.points for row in incumbent_utilities], dtype=float)
    candidate_ids = search_pool
    evaluations = []
    max_size = min(config.max_scored_slots, len(candidate_ids))
    min_size = min(config.min_scored_slots, max_size)
    books = [
        book
        for size in range(min_size, max_size + 1)
        for book in combinations(candidate_ids, size)
    ]
    total_books = len(books)
    logger.info(
        "book search: %d candidates, scored slots %d..%d, %d books over %d periods",
        len(candidate_ids),
        min_size,
        max_size,
        total_books,
        len(common_periods),
    )

    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    percentile_matrix = np.asarray(
        [
            [indexed[candidate_id][period].points_percentile for period in common_periods]
            for candidate_id in candidate_ids
        ],
        dtype=float,
    )
    bootstrap_weights, effective_block = moving_block_bootstrap_weights(
        len(common_periods),
        block_length=config.block_length,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    alpha = (1.0 - config.confidence) / 2.0
    chunk_size = 1024
    for chunk_start in range(0, total_books, chunk_size):
        chunk = books[chunk_start : chunk_start + chunk_size]
        book_percentiles = np.asarray(
            [
                percentile_matrix[
                    [candidate_index[candidate_id] for candidate_id in book_ids]
                ].mean(axis=0)
                for book_ids in chunk
            ],
            dtype=float,
        )
        book_points = -np.cos(book_percentiles * np.pi / 100.0) * 10.0
        deltas = book_points - incumbent_points
        estimates = deltas.mean(axis=1)
        bootstrap_means = deltas @ bootstrap_weights.T
        lowers = np.quantile(bootstrap_means, alpha, axis=1)
        uppers = np.quantile(bootstrap_means, 1.0 - alpha, axis=1)

        for offset, book_ids in enumerate(chunk):
            period_utilities = tuple(
                BookPeriodUtility(
                    period=period,
                    candidate_ids=book_ids,
                    points_percentile=float(book_percentiles[offset, period_index]),
                    points=cosine_points(float(book_percentiles[offset, period_index])),
                )
                for period_index, period in enumerate(common_periods)
            )
            evaluations.append(
                BookEvaluation(
                    candidate_ids=book_ids,
                    period_utilities=period_utilities,
                    delta_vs_incumbent=BootstrapMean(
                        estimate=float(estimates[offset]),
                        lower=float(lowers[offset]),
                        upper=float(uppers[offset]),
                        block_length=effective_block,
                        samples=config.bootstrap_samples,
                    ),
                )
            )
        evaluated = min(chunk_start + chunk_size, total_books)
        logger.info(
            "book search progress: %d/%d books",
            evaluated,
            total_books,
        )
    ranked = sorted(
        evaluations,
        key=lambda evaluation: (
            -evaluation.delta_vs_incumbent.lower,
            -evaluation.delta_vs_incumbent.estimate,
            len(evaluation.candidate_ids),
            tuple(map(str, evaluation.candidate_ids)),
        ),
    )
    if fixed_book:
        selected = next(
            evaluation
            for evaluation in evaluations
            if evaluation.candidate_ids == fixed_book
        )
    else:
        selected = ranked[0]
    selected_experimental: tuple[Hashable, ...] = ()
    if config.max_experimental_slots:
        selected_experimental = tuple(
            candidate_id
            for candidate_id in nominees
            if candidate_id not in selected.candidate_ids
        )[: config.max_experimental_slots]
        singles = [
            evaluation.candidate_ids[0]
            for evaluation in ranked
            if len(evaluation.candidate_ids) == 1
            and evaluation.candidate_ids[0] not in selected.candidate_ids
            and evaluation.candidate_ids[0] not in selected_experimental
        ]
        remaining = config.max_experimental_slots - len(selected_experimental)
        selected_experimental += tuple(singles[:remaining])
    return BookSelection(
        incumbent_ids=incumbent_ids,
        selected=selected,
        experimental_ids=selected_experimental,
        evaluations=tuple(ranked),
    )
