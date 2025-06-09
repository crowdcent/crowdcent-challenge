## Objective
The Hyperliquid Ranking Challenge requires participants to rank crypto assets on the Hyperliquid decentralized derivatives exchange by their expected relative returns over the next 10 and 30 days. The challenge universe comprises approximately 165-175 (and likely more in the future) liquid tokens on the Hyperliquid protocol. This universe may change periodically, with tokens added or removed to ensure it remains as actionable as possible. If a token does not have enough volume or liquidity, it will likely be removed from the universe.

```python
from crowdcent_challenge.client import CrowdCentClient
client = CrowdCentClient(challenge_slug="hyperliquid-ranking")
client.get_challenge() # Get more challenge details
```

## Inference data
- Inference Period Open: The internal pipeline *starts* at **~14:00&nbsp;UTC**. The file usually becomes available a few seconds to a few minutes later, once data quality checks pass.
- Inference Period Close: **4 hours after the actual release timestamp** (typically around 18:00&nbsp;UTC).

Each day, an inference dataset is released containing the universe of tokens for which predictions are required. The inference data contains features but has no targets as they do not exist at the time of your submission. Your predictions will be scored against resolving targets from real market data in the future.

!!! tip
    You do *not* need to use the features included in the inference data. You can use and are encouraged to use your own features. It may still be helpful to use our inference data and id mappings to use the same universe of assets.

To download inference data for the current period:
```python
client.download_inference_data("current")
```

| id      | eodhd_id              | feature_1_lag15 | feature_1_lag10 | feature_1_lag5 | feature_1_lag0 | ... |
|---------|----------------------|----------------|----------------|-----------------|-----------------|-----|
| BABY    | BABY32198-USD.CC     | 0.123          | 0.145          | 0.112           | 0.156           | ... |
| OM      | OM-USD.CC            | 0.456          | 0.423          | 0.467           | 0.401           | ... |
| IOTA    | IOTA-USD.CC          | 0.789          | 0.812          | 0.798           | 0.823           | ... |
| MOODENG | MOODENG33093-USD.CC  | 0.234          | 0.267          | 0.223           | 0.289           | ... |
| ENS     | ENS-USD.CC           | 0.567          | 0.534          | 0.578           | 0.512           | ... |

!!! note "Polling recommended"
    The inference file may not exist the very instant the clock strikes 14:00&nbsp;UTC. If you use the `download_inference_data` method with `release_date="current"` and default parameters, this is handled for you automatically. If you are interested in the details, or would like to poll/retry manually, you can use `client.wait_for_inference_data("inference_data.parquet")` from the Python client, which implements the retrying logic. If you are building your own client, you can gently poll the API until `GET /current` stops returning `404`. 

### Asset IDs
We currently provide `id` (the hyperliquid id) and `eodhd_id` (the id to download via EODHD) for each asset. You can request we include additional id mappings from other data vendors in the inference data if data licenses allow.

If you're using CrowdCent's training data to build your models, the inference data features will always match that of the *latest* training dataset version. The assets are generally the same as you would find in the training data, but new additions or removals are possible. For the inference data, we aim to track the listed and tradeable perps on Hyperliquid.

## Training data
The training dataset is created just to get you started. Simple models can be built with just the features and targets, but don't expect to win the challenge with them. We recommend building your own training datasets with sources like ccxt, eodhd, coingecko, or yfinance.

You can download our training data, including features and targets from [crowdcent.com/challenge/hyperliquid-ranking](https://crowdcent.com/challenge/hyperliquid-ranking) or via the CrowdCent client:
```python
client.download_training_data("latest")
```

| id      | eodhd_id             | date       | feature_1_lag15 | feature_1_lag10 | ... | feature_1_lag0 | target_10d | target_30d |
|---------|----------------------|------------|----------------|----------------|-----|------------------|------------|------------|
| BABY    | BABY32198-USD.CC     | 2024-03-20 | 0.123          | 0.145          | ... | 0.823            | 0.15       | 0.25       |
| OM      | OM-USD.CC            | 2024-03-20 | 0.456          | 0.423          | ... | 0.756            | 0.35       | 0.45       | 
| IOTA    | IOTA-USD.CC          | 2024-03-20 | 0.789          | 0.812          | ... | 0.923            | 0.55       | 0.65       |
| MOODENG | MOODENG33093-USD.CC  | 2024-03-20 | 0.234          | 0.267          | ... | 0.445            | 0.75       | 0.85       |
| ENS     | ENS-USD.CC           | 2024-03-20 | 0.567          | 0.534          | ... | 0.678            | 0.95       | 0.88       |

### Features
The training data contains 80 total features following the pattern `feature_{n}_lag{lag}`:

- 20 unique features (n = 1 to 20)
- 4 lag values per feature (0, 5, 10, 15 days)

Features with the same number represent the same metric at different time points. For example, `feature_1_lag0` through `feature_1_lag15` track the same underlying metric over time. This structure preserves temporal relationships, allowing you to build sequence models (LSTM, GRU, Transformer), identify trends/patterns across different time horizons, and engineer additional features based on temporal changes.

### Targets

Targets are the rankings of an asset's 10d and 30d forward relative returns (with a 1d lag). Targets do not currently take funding rate or any other factors (e.g. market cap, volume, etc.) into account. It's possible that the targets will be updated in the future to include such factors.

**Why the 1-day lag?**
For our purposes, the crypto universe has a close time of 24:00 UTC. At 14:00 UTC when predictions are made, only data through the previous day's 24:00 UTC close is available. The lag ensures predictions use only historical data while forecasting returns starting from tonight's close.

**Timeline:**
```
Day D-1: 24:00 UTC → Close price finalized (latest available data)
Day D:   14:00 UTC → Inference pipeline starts
         14:00-18:00 UTC → Inference period lasts 4 hours
         24:00 UTC → Prediction/Scoring period starts (Day D close)
Day D+10/30: 24:00 UTC → Prediction/Scoring period ends
```

- Predictions rank assets by expected performance over the next 10/30 days starting from tonight's close
- Rankings are relative. They say nothing about the expected absolute performance of an asset.

## Submitting predictions
Minimum of 80 ids from the inference data are required for a valid submission. The following columns are also required (no index):

- `id`: The id of the asset on Hyperliquid.
- `pred_10d`: A float between 0 and 1 representing the predicted rank for the 10-day horizon.
- `pred_30d`: A float between 0 and 1 representing the predicted rank for the 30-day horizon.


To submit predictions, you have 5 submission slots available. These can be used to submit multiple predictions for the same day and are defined by the `slot` parameter:
```python
client.submit_predictions(df=predictions_df, slot=1) # Submit a dataframe
client.submit_predictions(file_path="submission.parquet", slot=2) # or a parquet file
```


| id      | pred_10d | pred_30d |
|---------|----------|----------|
| BABY    | 0.2      | 0.3      |
| OM      | 0.4      | 0.5      |
| IOTA    | 0.1      | 0.2      |
| MOODENG | 1.0      | 1.0      |
| ENS     | 0.3      | 0.4      |

!!! Note
    If you are submitting a dataframe, `id` must be a column in the dataframe, *not* the index.
    If you are submitting a file, all submissions must be in parquet format.

## Scoring and Evaluation
Before scoring, for each prediction timeframe, ids are uniform ranked [0, 1], and any missing ids are filled with 0.5.

### Composite Score (Warm-up Phase)

During the initial *warm-up* phase of the challenge, the **overall daily score** is simply the average of four raw metrics: two horizons × two metric types:

Each metric is first converted into a **percentile rank** (0 → worst, 1 → best) relative to all submissions for the same day. The composite is then the simple average of those four percentiles:

$$\text{Overall Score} 
= \tfrac14\bigl( p\bigl(\text{NDCG@40}_{10\,d}\bigr)
            + p\bigl(\text{NDCG@40}_{30\,d}\bigr)
            + p\bigl(\rho_{\text{Spearman},\;10\,d}\bigr)
            + p\bigl(\rho_{\text{Spearman},\;30\,d}\bigr)\bigr)$$

where $p(\cdot)$ denotes your percentile across **all other submissions** on that day.

!!! warning "Minimum submissions for percentiles"
    Percentile ranks are **only calculated when five (5) or more valid submissions** are received for a given day. If fewer than five submissions are present, percentile-based metrics will be omitted for that day and the daily percentile will not contribute to any official scoring.

---

**Metrics:**

- [Symmetric Normalized Discounted Cumulative Gain (NDCG@40)](https://docs.crowdcent.com/scoring/#symmetric-normalized-discounted-cumulative-gain-symmetric-ndcgk)
When you read NDCG@40, think "how well did I rank the top 40 assets and how well did I rank the bottom 40 assets?" With ~180 tokens in the universe, k=40 represents approximately the top/bottom 20-25% of assets. This metric is symmetric, meaning it equally rewards:
  - **Top 40 identification**: Finding the tokens that will have the highest returns (for long positions)
  - **Bottom 40 identification**: Finding the tokens that will have the lowest returns (for short positions or avoidance)
  
  The logarithmic discount means getting the #1 ranked token correct is much more valuable than getting the #40 ranked token correct. A perfect NDCG@40 score of 1.0 means you perfectly ranked both tails of the distribution. This metric is particularly valuable for portfolio construction where you want to maximize exposure to the best performers while avoiding or shorting the worst.

- [Spearman Correlation](https://docs.crowdcent.com/scoring/#spearman-correlation)
Spearman correlation measures how well your predicted ranks align with the true ranks across the entire universe of ~180 tokens. Unlike NDCG which focuses on the extremes, Spearman treats all rank positions equally.

### Score Ranges and Percentile Rankings

**Score Ranges:**

- **NDCG@40**: Ranges from 0.0 (worst possible) to 1.0 (perfect ranking of top and bottom 40 assets)
- **Spearman Correlation**: Ranges from -1.0 (perfect inverse ranking) to 1.0 (perfect ranking), with 0.0 indicating random performance

**Why Are Scores Typically Low?**

Financial markets are characterized by extremely high noise-to-signal ratios. Seemingly "low" scores can be quite competitive in this domain. Additionally, market regimes shift over time, causing the distribution of achievable scores to fluctuate significantly.

**Percentile Rankings: Your Most Reliable Metric**

CrowdCent calculates **percentile rankings** that show where you stand relative to other participants. These percentiles are recalculated daily.

Tracking your percentile rank over time is often more informative than focusing on absolute scores, as it accounts for evolving competition and regime shifts that affect all participants.

!!! tip "Focus on Consistency"
    A model that consistently ranks in the 75th percentile across different market conditions is often more valuable than one that occasionally achieves top scores but performs poorly in other regimes.