import numpy as np
import narwhals as nw


def symmetric_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Calculates Symmetric Normalized Discounted Cumulative Gain at rank k.

    This metric evaluates the ranking quality for both the top-k highest
    predicted scores and the bottom-k lowest predicted scores, comparing
    them to the actual highest and lowest true values respectively.
    The final score is the average of the NDCG@k for the top and bottom.

    Args:
        y_true: Array of true target values. Must be in range [0, 1].
        y_pred: Array of predicted scores. Will be uniformly scaled to [0, 1].
        k: The rank cutoff for both top and bottom evaluation.

    Returns:
        The Symmetric NDCG@k score, ranging from 0 to 1.
    
    Raises:
        ValueError: If y_true contains values outside [0, 1].
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if len(y_true) == 0:
        return 0.0  # Or NaN, depending on desired behavior for empty input

    # Ensure y_true is in range [0, 1]
    if np.min(y_true) < 0 or np.max(y_true) > 1:
        raise ValueError("y_true values must be in range [0, 1].")
    
    # Uniformly scale y_pred to [0, 1]
    y_pred_min = np.min(y_pred)
    y_pred_max = np.max(y_pred)
    if y_pred_min == y_pred_max:
        # All predictions are the same - set to 0.5
        y_pred = np.full_like(y_pred, 0.5)
    else:
        y_pred = (y_pred - y_pred_min) / (y_pred_max - y_pred_min)

    # Reshape to 2D for consistency with helper functions
    y_true_2d = y_true.reshape(1, -1)
    y_pred_2d = y_pred.reshape(1, -1)

    # --- Top-k NDCG Calculation ---
    ndcg_top = _ndcg_sample_scores(y_true_2d, y_pred_2d, k=k, ignore_ties=False)[0]

    # --- Bottom-k NDCG Calculation ---
    # Transform to invert rankings while keeping values non-negative
    # This makes originally low values high, so NDCG will reward finding the originally lowest items
    y_true_inverted = 1 - y_true_2d
    y_pred_inverted = 1 - y_pred_2d
    
    ndcg_bottom = _ndcg_sample_scores(y_true_inverted, y_pred_inverted, k=k, ignore_ties=False)[0]

    # --- Average Top and Bottom ---
    symmetric_ndcg = (ndcg_top + ndcg_bottom) / 2.0

    return symmetric_ndcg

# -----------------------------------------------------------------------------
# Scikit-learn style DCG / NDCG implementation
# -----------------------------------------------------------------------------


def _tie_averaged_dcg(
    y_true: np.ndarray,
    y_score: np.ndarray,
    discount_cumsum: np.ndarray,
) -> float:
    """Average DCG over all possible permutations of tied ranks.

    This is a direct port of scikit-learn's private helper and is used when
    ``ignore_ties`` is *False* to obtain tie-robust DCG values.
    """
    # Identify tied groups by unique prediction values (descending order)
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    # Sum the relevance of each tied group
    ranked = np.zeros(len(counts), dtype=float)
    np.add.at(ranked, inv, y_true)
    # Replace gains by their average value inside each tied group
    ranked /= counts
    # Indices of the *last* element of each tied group once sorted
    groups = np.cumsum(counts) - 1
    # Sum of the discounts for all positions inside each tied group
    discount_sums = np.empty(len(counts), dtype=float)
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])

    return float((ranked * discount_sums).sum())


def _dcg_sample_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int | None = None,
    log_base: float = 2.0,
    ignore_ties: bool = False,
) -> np.ndarray:
    """Compute Discounted Cumulative Gain for each *sample*.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        Ground-truth relevance scores.
    y_score : ndarray of shape (n_samples, n_labels)
        Predicted scores that induce the ranking.
    k : int or None, default=None
        If given, only the highest-ranked ``k`` elements contribute to the sum.
    log_base : float, default=2
        Base for the logarithmic discount.
    ignore_ties : bool, default=False
        If *True*, assumes there are no ties in ``y_score`` for a faster
        computation (simply uses ``argsort``); otherwise, the metric is
        averaged over all permutations of tied groups following
        McSherry & Najork (2008).
    """
    # Make sure we are working with NumPy arrays of float64 for safety.
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            "y_true and y_score must have the same shape. "
            f"Got {y_true.shape} and {y_score.shape}."
        )

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)

    n_samples, n_labels = y_true.shape

    # Pre-compute the logarithmic discounts (1-indexed ranks)
    discount = 1.0 / (np.log(np.arange(n_labels) + 2) / np.log(log_base))
    if k is not None:
        discount[k:] = 0.0

    if ignore_ties:
        # Fast path: assume there are no ties, just sort scores descending.
        ranking = np.argsort(y_score)[:, ::-1]
        ranked_true = y_true[np.arange(n_samples)[:, None], ranking]
        return ranked_true.dot(discount)

    # Slower path that is robust to ties – compute sample-wise.
    discount_cumsum = np.cumsum(discount)
    return np.array(
        [
            _tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_score)
        ],
        dtype=float,
    )


def dcg_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int | None = None,
    log_base: float = 2.0,
    sample_weight: np.ndarray | None = None,
    ignore_ties: bool = False,
) -> float:
    """Compute the averaged Discounted Cumulative Gain (DCG).

    This is a near-verbatim re-implementation of
    ``sklearn.metrics.dcg_score`` that avoids requiring scikit-learn at
    runtime.  See the scikit-learn documentation for a full discussion of the
    metric.
    """
    gains = _dcg_sample_scores(
        y_true, y_score, k=k, log_base=log_base, ignore_ties=ignore_ties
    )
    if sample_weight is None:
        return float(np.mean(gains))
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.shape[0] != gains.shape[0]:
        raise ValueError(
            f"sample_weight has shape {sample_weight.shape} but there are {gains.shape[0]} samples."
        )
    return float(np.average(gains, weights=sample_weight))


def _ndcg_sample_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int | None = None,
    ignore_ties: bool = False,
) -> np.ndarray:
    """Compute sample-wise Normalized Discounted Cumulative Gain (NDCG)."""
    gain = _dcg_sample_scores(y_true, y_score, k=k, ignore_ties=ignore_ties)
    # Ideal DCG – using the true scores as perfect predictions.
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k=k, ignore_ties=True)
    # Handle samples with zero relevant items: define their NDCG as 0.
    mask = normalizing_gain > 0
    ndcg = np.zeros_like(gain, dtype=float)
    ndcg[mask] = gain[mask] / normalizing_gain[mask]
    return ndcg


def ndcg_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int | None = None,
    sample_weight: np.ndarray | None = None,
    ignore_ties: bool = False,
) -> float:
    """Compute the averaged Normalized Discounted Cumulative Gain (NDCG).

    The function mirrors ``sklearn.metrics.ndcg_score`` but is self-contained
    to avoid the heavy scikit-learn dependency at runtime.
    """
    y_true = np.asarray(y_true, dtype=float)
    if (y_true < 0).any():
        raise ValueError("ndcg_score should not be used on negative y_true values.")

    ndcg = _ndcg_sample_scores(y_true, y_score, k=k, ignore_ties=ignore_ties)
    if sample_weight is None:
        return float(np.mean(ndcg))
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.shape[0] != ndcg.shape[0]:
        raise ValueError(
            f"sample_weight has shape {sample_weight.shape} but there are {ndcg.shape[0]} samples."
        )
    return float(np.average(ndcg, weights=sample_weight))


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        y_true: Array of true target values
        y_pred: Array of predicted scores

    Returns:
        Spearman correlation coefficient, or 0.0 if calculation fails
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0

    try:
        corr, _ = spearmanr(y_pred, y_true)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def safe_metric(metric_func, y_true, y_pred, *args, **kwargs):
    """
    A wrapper that calls a metric function only if data is present, otherwise returns nan.
    """
    if not len(y_true) or not len(y_pred):
        return np.nan
    return metric_func(y_true, y_pred, *args, **kwargs)


def evaluate_hyperliquid_submission(
    y_true_10d: np.ndarray,
    y_pred_10d: np.ndarray,
    y_true_30d: np.ndarray,
    y_pred_30d: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a Hyperliquid ranking submission with correct metrics.

    Calculates:
    - Spearman correlation for 10d and 30d horizons (NOT Pearson)
    - Symmetric NDCG@40 for 10d and 30d horizons

    Args:
        y_true_10d: True target values for 10-day horizon
        y_pred_10d: Predicted scores for 10-day horizon
        y_true_30d: True target values for 30-day horizon
        y_pred_30d: Predicted scores for 30-day horizon

    Returns:
        Dict containing all individual metrics (no composite_score)
    """
    scores = {}

    # Spearman correlations
    scores["spearman_10d"] = safe_metric(spearman_correlation, y_true_10d, y_pred_10d)
    scores["spearman_30d"] = safe_metric(spearman_correlation, y_true_30d, y_pred_30d)

    # Symmetric NDCG@40
    scores["ndcg@40_10d"] = safe_metric(symmetric_ndcg_at_k, y_true_10d, y_pred_10d, k=40)
    scores["ndcg@40_30d"] = safe_metric(symmetric_ndcg_at_k, y_true_30d, y_pred_30d, k=40)

    return scores


@nw.narwhalify
def create_ranking_targets(
    df,
    horizons=[10, 30],
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_raw_returns: bool = False,
    drop_incomplete: bool = True,
):
    """
    Create cross-sectional ranking targets from price data.

    This function implements the standard methodology for creating ranking targets:
    1. Calculate forward returns for specified horizons
    2. Shift returns back by 1 day (so features on day D predict returns from D+1 to D+h+1)
    3. Rank returns cross-sectionally by date
    4. Normalize ranks to [0, 1] scale

    Trading Interpretation
    ----------------------
    Assuming close prices are at 24:00 UTC:
    - Features calculated from: Day D-1 close (24:00 UTC)
    - Predicting returns from: Day D close to Day D+h close
    - Trading window: Anytime during Day D

    For Hyperliquid Ranking Challenge:
    - Challenge opens: 14:00 UTC on Day D
    - Challenge closes: 18:00 UTC on Day D
    - You have a 4-hour window to submit predictions
    - Returns measured from Day D close (24:00 UTC) to Day D+h close

    Example Timeline (10-day horizon):
    - Sun 24:00 UTC: Features calculated from this close
    - Mon 14:00 UTC: Challenge opens, submit predictions
    - Mon 18:00 UTC: Challenge closes
    - Mon 24:00 UTC: Return period begins
    - Thu 24:00 UTC (10 days later): Return period ends

    Parameters
    ----------
    df : IntoFrameT
        Price data with columns for date, ticker, and price
    horizons : List[int], default [10, 30]
        Forward return horizons in days (e.g., [10, 30] for 10-day and 30-day targets)
    price_col : str, default "close"
        Column name for the price to use (typically "close" or "adjusted_close")
    date_col : str, default "date"
        Column name for dates
    ticker_col : str, default "ticker"
        Column name for ticker/asset identifiers
    return_raw_returns : bool, default False
        If True, also return the raw forward returns (not just ranks)
    drop_incomplete : bool, default True
        If True, drop rows where any target is null. If False, keep all rows
        and compute rankings only for non-null values within each group

    Returns
    -------
    IntoFrameT
        Original dataframe with added target columns:
        - target_{h}d: Normalized rank (0-1) for each horizon h
        - fwd_return_{h}d: Raw forward returns (if return_raw_returns=True)

    Example
    -------
    >>> # Basic usage for crowdcent-challenge
    >>> df_with_targets = create_ranking_targets(price_df)
    >>>
    >>> # Custom horizons
    >>> df_with_targets = create_ranking_targets(price_df, horizons=[5, 20, 60])
    >>>
    >>> # Keep raw returns for analysis
    >>> df_with_targets = create_ranking_targets(price_df, return_raw_returns=True)

    Notes
    -----
    - By default, drops rows where targets cannot be calculated (e.g., last h days)
    - Set drop_incomplete=False to keep partial targets (e.g., 1d target exists but 30d doesn't)
    - Rankings are done cross-sectionally within each date
    - Null values remain null in rankings and don't participate in rank calculation
    - This matches the standard scoring methodology for ranking challenges
    - Perfect for backtesting strategies before live submission
    """
    result_df = df.clone()

    # Calculate forward returns for each horizon
    for h in horizons:
        fwd_temp = f"_fwd_{h}d_temp"
        fwd_return = f"fwd_return_{h}d"

        result_df = (
            result_df
            # Calculate h-day forward return sitting on day D
            .with_columns(
                (
                    nw.col(price_col).shift(-h).over(ticker_col) / nw.col(price_col) - 1
                ).alias(fwd_temp)
            )
            # Shift back by 1 day so it sits on day D-1
            # This ensures features on D-1 predict returns from D to D+h
            .with_columns(nw.col(fwd_temp).shift(1).over(ticker_col).alias(fwd_return))
            .drop(fwd_temp)
        )

    # Optionally drop rows without targets
    if drop_incomplete:
        return_cols = [f"fwd_return_{h}d" for h in horizons]
        result_df = result_df.drop_nulls(subset=return_cols)

    # Create cross-sectional rankings
    for h in horizons:
        result_df = result_df.with_columns(
            (
                nw.col(f"fwd_return_{h}d").rank().over(date_col)
                / nw.col(date_col).count().over(date_col)
            ).alias(f"target_{h}d")
        )

    # Optionally drop raw returns
    if not return_raw_returns:
        result_df = result_df.drop(return_cols)

    return result_df
