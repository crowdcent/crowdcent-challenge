import numpy as np


def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """
    Calculates Discounted Cumulative Gain at rank k.

    Args:
        relevance_scores: An array of relevance scores, ordered by the ranking
                          being evaluated.
        k: The rank cutoff.

    Returns:
        The DCG@k score.
    """
    k = min(len(relevance_scores), k)
    if k == 0:
        return 0.0
    discounts = np.log2(np.arange(k) + 2)  # ranks 1..k -> log2(rank+1)
    return np.sum(relevance_scores[:k] / discounts)


def symmetric_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Calculates Symmetric Normalized Discounted Cumulative Gain at rank k.

    This metric evaluates the ranking quality for both the top-k highest
    predicted scores and the bottom-k lowest predicted scores, comparing
    them to the actual highest and lowest true values respectively.
    The final score is the average of the NDCG@k for the top and bottom.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted scores.
        k: The rank cutoff for both top and bottom evaluation.

    Returns:
        The Symmetric NDCG@k score, ranging from 0 to 1.
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

    # --- Top-k NDCG Calculation ---
    # Order by prediction (highest first)
    pred_order_desc = np.argsort(y_pred)[::-1]
    # Get true values in the predicted order
    true_relevance_in_pred_order_top = y_true[pred_order_desc]

    # Order by true value (highest first) for ideal ranking
    ideal_order_desc = np.argsort(y_true)[::-1]
    ideal_relevance_top = y_true[ideal_order_desc]

    # Calculate DCG and IDCG for the top
    dcg_top = dcg_at_k(true_relevance_in_pred_order_top, k)
    idcg_top = dcg_at_k(ideal_relevance_top, k)
    ndcg_top = dcg_top / idcg_top if idcg_top > 0 else 0.0

    # --- Bottom-k NDCG Calculation ---
    # Use negative true values as relevance for the bottom ranking.
    # Higher negative relevance means more truly negative y_true.
    y_true_neg_relevance = -y_true

    # Order by prediction (lowest first)
    pred_order_asc = np.argsort(y_pred)  # Ascending is default
    # Get negative true values in the predicted bottom order
    true_relevance_in_pred_order_bottom = y_true_neg_relevance[pred_order_asc]

    # Order by true value (lowest first) for ideal bottom ranking,
    # which means ordering by negative relevance (highest first)
    ideal_order_asc = np.argsort(y_true)  # lowest y_true first
    ideal_relevance_bottom = y_true_neg_relevance[ideal_order_asc]

    # Calculate DCG and IDCG for the bottom (using negative relevance)
    dcg_bottom = dcg_at_k(true_relevance_in_pred_order_bottom, k)
    idcg_bottom = dcg_at_k(ideal_relevance_bottom, k)
    # Note: if idcg_bottom is <= 0, it implies all *actual* bottom k true values
    # were >= 0, meaning no truly "negative" items were in the ideal bottom set.
    # In this case, ndcg_bottom should be 0.
    ndcg_bottom = dcg_bottom / idcg_bottom if idcg_bottom > 0 else 0.0

    # --- Combine ---
    symmetric_ndcg = (ndcg_top + ndcg_bottom) / 2.0

    # Clamp score between 0 and 1, as edge cases might theoretically yield slightly outside this.
    return max(0.0, min(1.0, symmetric_ndcg))
