import numpy as np
import polars as pl
import pytest

from crowdcent_challenge.scoring import (
    create_ranking_targets,
    dcg_score,
    symmetric_ndcg_at_k,
    evaluate_hyperliquid_submission,
    evaluate_hyperliquid_uniqueness,
    corr_to_meta,
    orthogonal_ic,
    unique_spearman,
    unique_ndcg,
    neutralize_predictions,
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


# --- Tests for corr_to_meta ------------------------------------------


def test_corr_to_meta_identical():
    """Identical predictions should have correlation of +1."""
    y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    meta_pred = y_pred.copy()
    assert np.isclose(corr_to_meta(y_pred, meta_pred), 1.0)


def test_corr_to_meta_anti_correlated():
    """Perfect anti-correlation should have correlation of -1."""
    y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    meta_pred = y_pred[::-1]  # Reversed = anti-correlated
    assert np.isclose(corr_to_meta(y_pred, meta_pred), -1.0)


def test_corr_to_meta_orthogonal():
    """Uncorrelated predictions should have correlation near 0."""
    np.random.seed(42)
    # Create two uncorrelated random vectors
    y_pred = np.random.rand(100)
    meta_pred = np.random.rand(100)
    score = corr_to_meta(y_pred, meta_pred)
    # Should be close to 0 since random vectors are ~uncorrelated
    assert -0.3 < score < 0.3


def test_corr_to_meta_partial_correlation():
    """Partially correlated predictions should have intermediate correlation."""
    np.random.seed(42)
    meta_pred = np.random.rand(100)
    # Create predictions that are 50% meta + 50% noise
    noise = np.random.rand(100)
    y_pred = 0.5 * meta_pred + 0.5 * noise
    score = corr_to_meta(y_pred, meta_pred)
    # Should be positive but not 1
    assert 0.3 < score < 0.9


def test_corr_to_meta_empty():
    """Empty arrays should return 0."""
    assert corr_to_meta(np.array([]), np.array([])) == 0.0


def test_corr_to_meta_length_mismatch():
    """Mismatched lengths should raise ValueError."""
    with pytest.raises(ValueError):
        corr_to_meta(np.array([1, 2, 3]), np.array([1, 2]))


# --- Tests for orthogonal_ic ------------------------------------------


def test_orthogonal_ic_perfect_unique_signal():
    """When unique signal perfectly predicts target, OIC should be high."""
    np.random.seed(42)
    n = 100
    
    # Meta model: random
    meta_pred = np.random.rand(n)
    
    # Target: completely orthogonal to meta
    # Create target from noise that's uncorrelated with meta
    y_true = np.random.rand(n)
    
    # User prediction: matches target (but has some meta exposure too)
    # After neutralizing to meta, the residual should still correlate with target
    y_pred = y_true.copy()
    
    score = orthogonal_ic(y_true, y_pred, meta_pred)
    # Should be positive since y_pred's orthogonal part predicts y_true
    assert score > 0.5


def test_orthogonal_ic_no_unique_signal():
    """When predictions are just meta, OIC should be near zero."""
    np.random.seed(42)
    n = 100
    
    meta_pred = np.random.rand(n)
    y_true = np.random.rand(n)  # Unrelated to meta
    y_pred = meta_pred.copy()  # Predictions are exactly meta
    
    score = orthogonal_ic(y_true, y_pred, meta_pred)
    # After neutralizing, residual is ~0, so correlation is undefined/near 0
    assert abs(score) < 0.3


def test_orthogonal_ic_wrong_unique_signal():
    """When unique signal is anti-correlated with target, OIC should be negative."""
    np.random.seed(42)
    n = 100
    
    meta_pred = np.random.rand(n)
    y_true = np.random.rand(n)
    # Predictions are opposite of target
    y_pred = 1 - y_true
    
    score = orthogonal_ic(y_true, y_pred, meta_pred)
    # Should be negative since unique part predicts incorrectly
    assert score < -0.5


def test_orthogonal_ic_constant_meta():
    """When meta is constant, all of y_pred is orthogonal."""
    np.random.seed(42)
    n = 100
    
    meta_pred = np.full(n, 0.5)  # Constant meta
    y_true = np.random.rand(n)
    y_pred = y_true + np.random.rand(n) * 0.3  # Correlated with target
    
    score = orthogonal_ic(y_true, y_pred, meta_pred)
    # Should be positive - all of y_pred is "orthogonal" to constant meta
    assert score > 0.3


def test_orthogonal_ic_empty():
    """Empty arrays should return 0."""
    assert orthogonal_ic(np.array([]), np.array([]), np.array([])) == 0.0


def test_orthogonal_ic_length_mismatch():
    """Mismatched lengths should raise ValueError."""
    with pytest.raises(ValueError):
        orthogonal_ic(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2]))


def test_orthogonal_ic_range():
    """OIC (Spearman) should always be in [-1, 1]."""
    np.random.seed(42)
    for _ in range(10):
        n = 50
        y_true = np.random.rand(n)
        y_pred = np.random.rand(n)
        meta_pred = np.random.rand(n)
        score = orthogonal_ic(y_true, y_pred, meta_pred)
        assert -1 <= score <= 1


# --- Tests for neutralize_predictions ------------------------------------------


def test_neutralize_predictions_removes_meta_exposure():
    """After neutralization, residuals should be uncorrelated with meta."""
    from scipy.stats import pearsonr
    np.random.seed(42)
    n = 100

    meta_pred = np.random.rand(n)
    # Create predictions highly correlated with meta
    y_pred = meta_pred * 0.8 + np.random.rand(n) * 0.2

    residual = neutralize_predictions(y_pred, meta_pred)
    
    # Residual should be ~uncorrelated with meta
    corr, _ = pearsonr(residual, meta_pred)
    assert abs(corr) < 0.1


def test_neutralize_predictions_constant_meta():
    """When meta is constant, residual is mean-centered (intercept absorbs mean)."""
    y_pred = np.array([0.1, 0.5, 0.9])
    meta_pred = np.array([0.5, 0.5, 0.5])
    
    residual = neutralize_predictions(y_pred, meta_pred)
    # Ranks are preserved (same ordering as y_pred)
    np.testing.assert_array_almost_equal(residual, y_pred - np.mean(y_pred))


# --- Tests for unique_spearman ------------------------------------------


def test_unique_spearman_matches_orthogonal_ic_default():
    """unique_spearman should match orthogonal_ic with default metric."""
    np.random.seed(42)
    n = 50
    y_true = np.random.rand(n)
    y_pred = np.random.rand(n)
    meta_pred = np.random.rand(n)
    
    score1 = unique_spearman(y_true, y_pred, meta_pred)
    score2 = orthogonal_ic(y_true, y_pred, meta_pred)
    
    assert np.isclose(score1, score2)


# --- Tests for unique_ndcg ------------------------------------------


def test_unique_ndcg_range():
    """Unique NDCG should be in [0, 1]."""
    np.random.seed(42)
    for _ in range(10):
        n = 100
        y_true = np.random.rand(n)
        y_pred = np.random.rand(n)
        meta_pred = np.random.rand(n)
        score = unique_ndcg(y_true, y_pred, meta_pred, k=40)
        assert 0 <= score <= 1


def test_unique_ndcg_perfect_unique_signal():
    """When unique signal perfectly identifies top/bottom, unique NDCG should be high."""
    np.random.seed(42)
    n = 100

    # Meta: random
    meta_pred = np.random.rand(n)
    
    # Target: distinct from meta
    y_true = np.random.rand(n)
    
    # User prediction: matches target well
    y_pred = y_true + np.random.rand(n) * 0.1
    
    score = unique_ndcg(y_true, y_pred, meta_pred, k=40)
    # Should be reasonably high since y_pred's unique part predicts y_true
    assert score > 0.5


def test_unique_ndcg_vs_unique_spearman_different():
    """Unique NDCG and Unique Spearman should generally give different values."""
    np.random.seed(42)
    n = 100
    y_true = np.random.rand(n)
    y_pred = np.random.rand(n)
    meta_pred = np.random.rand(n)
    
    spearman_score = unique_spearman(y_true, y_pred, meta_pred)
    ndcg_score = unique_ndcg(y_true, y_pred, meta_pred, k=40)
    
    # They measure different things, so shouldn't be identical
    # (though could be close by chance)
    # Just verify both compute without error and are in valid ranges
    assert -1 <= spearman_score <= 1
    assert 0 <= ndcg_score <= 1


# --- Tests for evaluate_hyperliquid_uniqueness ------------------------------------------


def test_evaluate_hyperliquid_uniqueness_keys_and_ranges():
    """Returns all 6 expected keys with values in valid ranges."""
    np.random.seed(42)
    n = 100
    from scipy.stats import rankdata

    y_true_10d = rankdata(np.random.randn(n)) / n
    y_pred_10d = rankdata(np.random.randn(n)) / n
    meta_pred_10d = rankdata(np.random.randn(n)) / n
    y_true_30d = rankdata(np.random.randn(n)) / n
    y_pred_30d = rankdata(np.random.randn(n)) / n
    meta_pred_30d = rankdata(np.random.randn(n)) / n

    scores = evaluate_hyperliquid_uniqueness(
        y_true_10d, y_pred_10d, meta_pred_10d,
        y_true_30d, y_pred_30d, meta_pred_30d,
    )

    assert set(scores.keys()) == {
        "corr_to_meta_10d",
        "corr_to_meta_30d",
        "unique_spearman_10d",
        "unique_spearman_30d",
        "unique_ndcg@40_10d",
        "unique_ndcg@40_30d",
    }

    for key, value in scores.items():
        assert isinstance(value, float), f"{key} is not float"
        if "corr_to_meta" in key or "unique_spearman" in key:
            assert -1 <= value <= 1, f"{key}={value} out of [-1, 1]"
        else:
            assert 0 <= value <= 1, f"{key}={value} out of [0, 1]"


def test_evaluate_hyperliquid_uniqueness_empty():
    """Empty arrays should return nan for all metrics."""
    empty = np.array([])
    scores = evaluate_hyperliquid_uniqueness(empty, empty, empty, empty, empty, empty)

    assert len(scores) == 6
    for key, value in scores.items():
        assert np.isnan(value), f"{key} should be nan for empty input"


# --- Tests for create_ranking_targets ------------------------------------------


def test_create_ranking_targets_forward_returns_order_invariant():
    """
    Forward returns must be computed in chronological order within each ticker,
    regardless of whether the input is sorted ascending or descending by date.
    """
    df = pl.DataFrame(
        {
            "ticker": ["A"] * 5 + ["B"] * 5,
            "date": [1, 2, 3, 4, 5] * 2,
            # A: +10% per step, B: -10% per step
            "close": [
                100.0,
                110.0,
                121.0,
                133.1,
                146.41,
                200.0,
                180.0,
                162.0,
                145.8,
                131.22,
            ],
        }
    )

    df_asc = df.sort(["ticker", "date"])
    df_desc = df.sort(["ticker", "date"], descending=[False, True])

    out_asc = create_ranking_targets(
        df_asc, horizons=[1], return_raw_returns=True, drop_incomplete=False
    ).sort(["ticker", "date"])
    out_desc = create_ranking_targets(
        df_desc, horizons=[1], return_raw_returns=True, drop_incomplete=False
    ).sort(["ticker", "date"])

    # fwd_return_1d at date t = return from t+1 to t+2 (because features at t are used on t+1)
    expected_a = [0.1, 0.1, 0.1, None, None]
    expected_b = [-0.1, -0.1, -0.1, None, None]

    a = out_asc.filter(pl.col("ticker") == "A")["fwd_return_1d"].to_list()
    b = out_asc.filter(pl.col("ticker") == "B")["fwd_return_1d"].to_list()
    assert a[:3] == pytest.approx(expected_a[:3])
    assert b[:3] == pytest.approx(expected_b[:3])
    assert a[3:] == expected_a[3:]
    assert b[3:] == expected_b[3:]

    # Desc vs asc should match (order-invariant)
    np.testing.assert_allclose(
        out_asc["fwd_return_1d"].fill_null(np.nan).to_numpy(),
        out_desc["fwd_return_1d"].fill_null(np.nan).to_numpy(),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )


def test_create_ranking_targets_preserves_input_row_order():
    """create_ranking_targets should return data sorted by ticker/date."""
    df_desc = pl.DataFrame(
        {
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "date": [3, 2, 1, 3, 2, 1],
            "close": [121.0, 110.0, 100.0, 162.0, 180.0, 200.0],
        }
    )

    out = create_ranking_targets(
        df_desc, horizons=[1], return_raw_returns=True, drop_incomplete=False
    )

    expected = df_desc.sort(["ticker", "date"])
    assert out["ticker"].to_list() == expected["ticker"].to_list()
    assert out["date"].to_list() == expected["date"].to_list()
