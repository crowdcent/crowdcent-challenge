All scoring functions used in the CrowdCent Challenge code directly from this repository.

```python
from crowdcent_challenge.scoring import *
```

## Symmetric Normalized Discounted Cumulative Gain (Symmetric NDCG@k)

One of the key metrics used in some challenges is `symmetric_ndcg_at_k`.

**Concept:**

Standard **Normalized Discounted Cumulative Gain (NDCG@k)** is a common metric used to evaluate how well a model ranks items. It assesses the quality of the top *k* predictions by:
1.  Giving higher scores for ranking truly relevant items higher.
2.  Applying a logarithmic discount to items ranked lower (meaning relevance at rank 1 is more important than relevance at rank 10).
3.  Normalizing the score by the best possible ranking (IDCG) to get a value between 0 and 1.

However, standard NDCG@k only focuses on the *top* performers. In finance, identifying the *worst* performers (lowest true values) can be just as important as identifying the best.

**Symmetric NDCG@k** addresses this by evaluating ranking performance at *both ends* of the spectrum:

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