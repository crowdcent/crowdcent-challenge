import numpy as np
import pytest

from crowdcent_challenge.scoring import (
    dcg_score,
    symmetric_ndcg_at_k,
    evaluate_hyperliquid_submission,
)


# --- Tests for dcg_score -------------------------------


def test_dcg_score_basic():
    """Validate DCG against a hand-computed example."""
    # For this test, we have relevance scores already in ranking order
    relevance = np.array([3, 2, 3, 0, 1, 2])
    k = 6

    # Create scores that will produce this ranking (highest to lowest)
    n = len(relevance)
    y_score = np.arange(n)[::-1].astype(float)  # [5, 4, 3, 2, 1, 0]
    y_true = relevance  # The relevance values

    # Manually compute expected DCG using the definition
    discounts = np.log2(np.arange(k) + 2)  # 1-indexed ranks => log2(rank+1)
    expected = np.sum(relevance[:k] / discounts)

    assert np.isclose(dcg_score(y_true, y_score, k=k), expected)


def test_dcg_score_handles_k_greater_than_length():
    """If k exceeds len(relevance) no error should occur and result is correct."""
    relevance = np.array([1, 0, 2])
    k = 10  # larger than len(relevance)

    # Create scores that will produce the given ranking order
    n = len(relevance)
    y_score = np.arange(n)[::-1].astype(float)  # [2, 1, 0]
    y_true = relevance

    discounts = np.log2(np.arange(len(relevance)) + 2)
    expected = np.sum(relevance / discounts)

    assert np.isclose(dcg_score(y_true, y_score, k=k), expected)


def test_dcg_score_k_zero():
    """k==0 should handle gracefully."""
    y_true = np.array([1, 2, 3])
    y_score = np.array([0.1, 0.2, 0.3])
    # When k=0, no items contribute to the sum, but dcg_score doesn't
    # explicitly handle k=0, so we expect 0.0 from the discount[k:] = 0 logic
    result = dcg_score(y_true, y_score, k=0)
    assert result == 0.0


def test_dcg_score_with_ties():
    """Test DCG with tied scores."""
    y_true = np.array([3, 2, 1])
    y_score = np.array([1, 1, 0])  # First two items tied

    # With ties, the average relevance of tied items is used
    result = dcg_score(y_true, y_score, k=2, ignore_ties=False)
    # Expected: average of (3,2) = 2.5 for both positions 1 and 2
    # DCG = 2.5/log2(2) + 2.5/log2(3)
    expected = 2.5 / np.log2(2) + 2.5 / np.log2(3)
    assert np.isclose(result, expected)


# --- Tests for symmetric_ndcg_at_k ------------------------------------------


def test_symmetric_ndcg_perfect_ranking_returns_one():
    """Perfect prediction should yield a score of exactly 1.0."""
    # Use normalized ranks in [0, 1]
    y_true = np.array([1.0, 0.857, 0.714, 0.571, 0.429, 0.286, 0.143])
    y_pred = y_true.copy()  # identical ranking
    k = 3
    assert np.isclose(symmetric_ndcg_at_k(y_true, y_pred, k), 1.0)


def test_symmetric_ndcg_worst_ranking_near_zero():
    """Completely reversed ranking should produce a score close to 0."""
    # Use normalized ranks in [0, 1]
    y_true = np.array([1.0, 0.833, 0.667, 0.5, 0.333, 0.167, 0.0])
    y_pred = y_true[::-1]  # reverse the order
    k = 3
    score = symmetric_ndcg_at_k(y_true, y_pred, k)
    assert score < 0.2  # low score for worst ranking


def test_symmetric_ndcg_length_mismatch_raises():
    """y_true and y_pred with unequal length should raise ValueError."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        symmetric_ndcg_at_k(y_true, y_pred, k=2)


def test_symmetric_ndcg_empty_input_returns_zero():
    """Empty inputs should return 0.0 as specified in docstring."""
    y_true = np.array([])
    y_pred = np.array([])
    assert symmetric_ndcg_at_k(y_true, y_pred, k=1) == 0.0


def test_evaluate_hyperliquid_submission():
    """Test the main evaluation function."""
    n = 100
    np.random.seed(42)

    # Create correlated predictions with raw values
    y_true_10d_raw = np.random.randn(n)
    y_pred_10d_raw = y_true_10d_raw + np.random.randn(n) * 0.5

    y_true_30d_raw = np.random.randn(n)
    y_pred_30d_raw = y_true_30d_raw + np.random.randn(n) * 0.5
    
    # Convert to normalized ranks in [0, 1]
    from scipy.stats import rankdata
    y_true_10d = rankdata(y_true_10d_raw) / n
    y_pred_10d = rankdata(y_pred_10d_raw) / n
    
    y_true_30d = rankdata(y_true_30d_raw) / n
    y_pred_30d = rankdata(y_pred_30d_raw) / n

    scores = evaluate_hyperliquid_submission(
        y_true_10d, y_pred_10d, y_true_30d, y_pred_30d
    )

    # Check all expected keys exist
    assert set(scores.keys()) == {
        "spearman_10d",
        "spearman_30d",
        "ndcg@40_10d",
        "ndcg@40_30d",
    }

    # Check value ranges
    for key, value in scores.items():
        if "spearman" in key:
            assert -1 <= value <= 1
        else:  # NDCG scores
            assert 0 <= value <= 1
