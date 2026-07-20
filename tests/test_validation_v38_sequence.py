"""PreparedDataset sequence field tests."""

from datetime import date

import numpy as np
import polars as pl
from crowdcent_challenge.validation.training.features import SequenceFeatureSchema
from crowdcent_challenge.validation.v38 import PreparedDataset, prepared_dataset_from_frame


def test_prepared_dataset_take_preserves_sequence_fields():
    schema = SequenceFeatureSchema(
        ordered_columns=("feature_0_lag0", "feature_1_lag0"),
        feature_ids=(0, 1),
        lag_windows=(0,),
    )
    dataset = PreparedDataset(
        periods=(date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2)),
        ids=("BTC", "ETH", "BTC"),
        targets=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float),
        features=np.ones((3, 2), dtype=np.float32),
        sequence_features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        sequence_schema=schema,
    )
    sliced = dataset.take(dataset.mask_between(date(2026, 1, 1), date(2026, 1, 1)))
    assert sliced.sequence_schema is schema
    assert sliced.sequence_features.shape == (2, 2)
    assert sliced.sequence_features[0, 0] == 1.0
    assert sliced.sequence_features[1, 0] == 3.0


def test_prepared_dataset_from_frame_builds_sequence_grid():
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "id": ["BTC", "ETH"],
            "target_10d": [0.2, 0.8],
            "target_30d": [0.3, 0.7],
            "feature_0_lag0": [0.1, 0.9],
            "feature_1_lag0": [0.2, 0.8],
            "feature_0_lag5": [0.3, 0.7],
            "feature_1_lag5": [0.4, 0.6],
            "feature_0_lag10": [0.5, 0.5],
            "feature_1_lag10": [0.6, 0.4],
            "feature_0_lag15": [0.7, 0.3],
            "feature_1_lag15": [0.8, 0.2],
        }
    )
    dataset = prepared_dataset_from_frame(frame)
    assert dataset.sequence_schema is not None
    assert dataset.sequence_schema.n_features_per_timestep == 2
    assert dataset.sequence_schema.lag_windows == (0, 5, 10, 15)
    assert dataset.sequence_features.shape == (2, 8)
