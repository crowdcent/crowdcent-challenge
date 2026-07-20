"""Optional full-fidelity training components for research environments."""

from ._optional import OptionalTrainingDependencyError
from .features import (
    SequenceFeatureSchema,
    engineer_cross_sectional_ranks,
    engineer_lag_differences,
    sequence_feature_schema,
    sequence_matrix,
    validate_prepared_frame,
)
from .models import (
    NotebookModelConfig,
    SequenceFit,
    XGBTuning,
    fit_sequence_model,
    fit_xgb_family,
    mean_daily_spearman,
    predict_xgb_family,
    recency_weights,
    refit_sequence_model,
    sequence_model_builders,
    tune_xgb_family,
)
from .recipe import (
    FiveSlotPredictions,
    HorizonRecipe,
    RawHorizonPredictions,
    StackWeights,
    build_five_slot_predictions,
    midpoint_rank,
    rank_blend,
)
from .runner import (
    NestedCalibrationConfig,
    NestedWalkForwardFold,
    build_nested_walk_forward_folds,
)

__all__ = [
    "FiveSlotPredictions",
    "HorizonRecipe",
    "NotebookModelConfig",
    "NestedCalibrationConfig",
    "NestedWalkForwardFold",
    "OptionalTrainingDependencyError",
    "RawHorizonPredictions",
    "SequenceFeatureSchema",
    "SequenceFit",
    "StackWeights",
    "XGBTuning",
    "build_five_slot_predictions",
    "build_nested_walk_forward_folds",
    "engineer_cross_sectional_ranks",
    "engineer_lag_differences",
    "fit_sequence_model",
    "fit_xgb_family",
    "mean_daily_spearman",
    "midpoint_rank",
    "predict_xgb_family",
    "rank_blend",
    "recency_weights",
    "refit_sequence_model",
    "sequence_feature_schema",
    "sequence_matrix",
    "sequence_model_builders",
    "tune_xgb_family",
    "validate_prepared_frame",
]
