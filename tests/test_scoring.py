import numpy as np
import pytest

from crowdcent_challenge.scoring import dcg_at_k, symmetric_ndcg_at_k


# --- Tests for dcg_at_k -----------------------------------------------------


def test_dcg_at_k_basic():
    """Validate DCG against a hand-computed example."""
    relevance = np.array([3, 2, 3, 0, 1, 2])
    k = 6
    # Manually compute expected DCG using the definition
    discounts = np.log2(np.arange(k) + 2)  # 1-indexed ranks => log2(rank+1)
    expected = np.sum(relevance[:k] / discounts)

    assert np.isclose(dcg_at_k(relevance, k), expected)


def test_dcg_at_k_handles_k_greater_than_length():
    """If k exceeds len(relevance) no error should occur and result is correct."""
    relevance = np.array([1, 0, 2])
    k = 10  # larger than len(relevance)
    discounts = np.log2(np.arange(len(relevance)) + 2)
    expected = np.sum(relevance / discounts)

    assert np.isclose(dcg_at_k(relevance, k), expected)


def test_dcg_at_k_k_zero_returns_zero():
    """k==0 should return 0.0 by definition."""
    relevance = np.array([1, 2, 3])
    assert dcg_at_k(relevance, 0) == 0.0


# --- Tests for symmetric_ndcg_at_k ------------------------------------------


def test_symmetric_ndcg_perfect_ranking_returns_one():
    """Perfect prediction should yield a score of exactly 1.0."""
    y_true = np.array([10, 9, 8, 2, 0, -1, -5])
    y_pred = y_true.copy()  # identical ranking
    k = 3
    assert np.isclose(symmetric_ndcg_at_k(y_true, y_pred, k), 1.0)


def test_symmetric_ndcg_worst_ranking_near_zero():
    """Completely reversed ranking should produce a score close to 0."""
    y_true = np.array([3, 2, 1, 0, -1, -2, -3])
    y_pred = -y_true  # reverse the order
    k = 3
    score = symmetric_ndcg_at_k(y_true, y_pred, k)
    assert score < 0.1  # near zero (exact 0 might not happen due to ties/fp)


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
