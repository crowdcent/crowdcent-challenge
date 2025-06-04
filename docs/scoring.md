All scoring functions used in the CrowdCent Challenge can be found in the `crowdcent_challenge.scoring` module.

```python
from crowdcent_challenge.scoring import *
```

## Symmetric NDCG@k

One of the key metrics used in some challenges is Symmetric Normalized Discounted Cumulative Gain (Symmetric NDCG@k).

**Concept:**

**Normalized Discounted Cumulative Gain (NDCG@k)** is a metric used to evaluate how well a model ranks items. It assesses the quality of the top *k* predictions by:

1.  Giving higher scores for ranking truly relevant items higher.
2.  Applying a logarithmic discount to items ranked lower (meaning relevance at rank 1 is more important than relevance at rank 10).
3.  Normalizing the score by the best possible ranking (IDCG) to get a value between 0 and 1.

However, this commonly used metric only focuses on the *top* items in a list. In finance, identifying the *worst* performers (lowest true values) can be just as important as identifying the best.

Our metric of **Symmetric** NDCG@k addresses this by evaluating ranking performance at *both ends* of the list:

1.  **Top Performance:** It calculates the standard `NDCG@k` based on your predicted scores (`y_pred`) compared to the actual true values (`y_true`). This measures how well you identify the items with the highest true values.
2.  **Bottom Performance:** It calculates another `NDCG@k` focused on the lowest ranks. It does this by:
    *   Ranking items based on your *lowest* predicted scores.
    *   Using the *negative* of the true values (`-y_true`) as relevance. This makes the most negative true values the most "relevant" for this bottom-ranking task.
    *   Calculating `NDCG@k` for how well your lowest predictions match the items with the lowest true values.
3.  **Averaging:** The final `symmetric_ndcg_at_k` score is the simple average of the Top NDCG@k and the Bottom NDCG@k. `(NDCG_top + NDCG_bottom) / 2`.

**Interpretation:**

*   A score closer to 1 indicates the model is excellent at identifying both the top *k* best and bottom *k* worst items according to their true values.
*   A score closer to 0 indicates poor performance at identifying the extremes.

**Usage:**

```python
from crowdcent_challenge.scoring import symmetric_ndcg_at_k
import numpy as np

# Example data
y_true = np.array([0.1, -0.2, 0.5, -0.1, 0.3])
y_pred = np.array([0.2, -0.1, 0.6, 0.0, 0.4])
k = 3

score = symmetric_ndcg_at_k(y_true, y_pred, k)
print(f"Symmetric NDCG@{k}: {score:.4f}")
```

This metric provides a more holistic view of ranking performance when both high and low extremes are important.

## Spearman Correlation

Spearman's rank correlation coefficient is a non-parametric measure of rank correlation that assesses how well the relationship between two variables can be described using a monotonic function.

**Concept:**

In the context of ranking challenges, **Spearman correlation** measures how well your predicted rankings align with the true rankings. Unlike Pearson correlation (which measures linear relationships), Spearman correlation:

1. **Works with ranks**: It converts both predicted and true values to ranks before computing correlation
2. **Captures monotonic relationships**: Perfect Spearman correlation (±1) means perfect monotonic relationship, even if not linear
3. **Robust to outliers**: Since it uses ranks rather than raw values, extreme values have less influence

**Calculation:**

The Spearman correlation coefficient (ρ) is calculated as:
- First, convert both `y_true` and `y_pred` to ranks
- Then calculate the Pearson correlation coefficient on these ranks
- Formula: ρ = 1 - (6 × Σd²) / (n × (n² - 1)), where d is the difference between paired ranks

**Interpretation:**

- **ρ = 1**: Perfect positive correlation (your rankings perfectly match the true rankings)
- **ρ = 0**: No correlation (your rankings are unrelated to the true rankings)  
- **ρ = -1**: Perfect negative correlation (your rankings are exactly reversed)

**Usage:**

```python
from scipy.stats import spearmanr
import numpy as np

# Example data
y_true = np.array([1.0, 0.5, 0.3, 0.2, 0.1])  # True values (will be ranked)
y_pred = np.array([0.9, 0.6, 0.25, 0.22, 0.05])    # Predicted values (will be ranked)

# Calculate Spearman correlation
correlation, p_value = spearmanr(y_true, y_pred)
print(f"Spearman Correlation: {correlation:.4f}")
```

**Why Spearman for Ranking Tasks?**

Spearman correlation is particularly suited for ranking challenges because:

1. **Focuses on order**: In ranking tasks, we care about getting the order right, not exact values
3. **Complements NDCG**: While NDCG focuses on top/bottom performance, Spearman evaluates the entire ranking