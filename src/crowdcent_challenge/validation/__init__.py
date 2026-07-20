"""Era-aware validation tools for the Hyperliquid Ranking challenge."""

from .atomic import (
    META_DIAGNOSTICS,
    RAW_METRICS,
    UNIQUE_METRICS,
    AtomicPeriodScore,
    InsufficientAssetsError,
    aggregate_objective,
    book_objective,
    score_atomic_period,
    uniform_rank,
)
from .book import (
    LiveSlotPercentile,
    ScoredSubsetResult,
    evaluate_scored_subsets,
    performance_adjustment,
)
from .diagnostics import (
    SelfInclusionDiagnostic,
    UniverseChurn,
    self_inclusion_diagnostic,
    universe_churn,
    weighted_meta_model,
)
from .eras import (
    HYPERLIQUID_RANKING_ERAS,
    ChallengeEras,
    PointsObjective,
)
from .pit import (
    MetaPurpose,
    latest_available_meta_release,
    meta_coverage,
    meta_release_cutoff,
)
from .promotion import (
    PromotionDecision,
    PromotionPolicy,
    ShadowPercentile,
    evaluate_shadow_promotion,
)
from .walkforward import (
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_folds,
)

__all__ = [
    "META_DIAGNOSTICS",
    "RAW_METRICS",
    "UNIQUE_METRICS",
    "AtomicPeriodScore",
    "ChallengeEras",
    "HYPERLIQUID_RANKING_ERAS",
    "InsufficientAssetsError",
    "LiveSlotPercentile",
    "MetaPurpose",
    "PointsObjective",
    "PromotionDecision",
    "PromotionPolicy",
    "SelfInclusionDiagnostic",
    "ScoredSubsetResult",
    "ShadowPercentile",
    "UniverseChurn",
    "WalkForwardConfig",
    "WalkForwardFold",
    "aggregate_objective",
    "book_objective",
    "build_walk_forward_folds",
    "evaluate_shadow_promotion",
    "evaluate_scored_subsets",
    "latest_available_meta_release",
    "meta_coverage",
    "meta_release_cutoff",
    "performance_adjustment",
    "score_atomic_period",
    "self_inclusion_diagnostic",
    "uniform_rank",
    "universe_churn",
    "weighted_meta_model",
]
