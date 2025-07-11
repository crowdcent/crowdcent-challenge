!!! example "Zero to Submission in 20 Seconds"
    Go from setup to your first prediction submission with our [end-to-end tutorial notebook](tutorials/hyperliquid-end-to-end.ipynb).

## Objective
The Hyperliquid Ranking Challenge requires participants to rank crypto assets on the Hyperliquid decentralized derivatives exchange by their expected relative returns over the next 10 and 30 days. The challenge universe comprises approximately 165-175 (and likely more in the future) liquid tokens on the Hyperliquid protocol. This universe may change periodically, with tokens added or removed to ensure it remains as actionable as possible. If a token does not have enough volume or liquidity, it will likely be removed from the universe.

```python
from crowdcent_challenge import ChallengeClient
client = ChallengeClient(challenge_slug="hyperliquid-ranking")
client.get_challenge() # Get more challenge details
```

## Training data
The training dataset is created just to get you started. Simple models can be built with just the features and targets, but don't expect to win the challenge with them. We recommend building your own training datasets with sources like ccxt, eodhd, coingecko, or yfinance.

You can download our training data, including features and targets from [crowdcent.com/challenge/hyperliquid-ranking](https://crowdcent.com/challenge/hyperliquid-ranking) or via the CrowdCent client:
```python
client.download_training_dataset("latest")
```

| id      | eodhd_id             | date       | feature_1_lag15 | feature_1_lag10 | ... | feature_1_lag0 | target_10d | target_30d |
|---------|----------------------|------------|----------------|----------------|-----|------------------|------------|------------|
| BABY    | BABY32198-USD.CC     | 2024-03-20 | 0.123          | 0.145          | ... | 0.823            | 0.15       | 0.25       |
| OM      | OM-USD.CC            | 2024-03-20 | 0.456          | 0.423          | ... | 0.756            | 0.35       | 0.45       | 
| IOTA    | IOTA-USD.CC          | 2024-03-20 | 0.789          | 0.812          | ... | 0.923            | 0.55       | 0.65       |
| MOODENG | MOODENG33093-USD.CC  | 2024-03-20 | 0.234          | 0.267          | ... | 0.445            | 0.75       | 0.85       |
| ENS     | ENS-USD.CC           | 2024-03-20 | 0.567          | 0.534          | ... | 0.678            | 0.95       | 0.88       |

### Asset IDs
We currently provide `id` (the hyperliquid id) and `eodhd_id` (the id to download via EODHD) for each asset. You can request we include additional id mappings from other data vendors in the inference data if data licenses allow.

If you're using CrowdCent's training data to build your models, the inference data features will always match that of the *latest* training dataset version. The assets are generally the same as you would find in the training data, but new additions or removals are possible. For the inference data, we aim to track the listed and tradeable perps on Hyperliquid.

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

## Inference data

- Inference Period Open: The internal pipeline *starts* at **14:00&nbsp;UTC**. The file usually becomes available a few seconds to a few minutes later, once data quality checks pass.
- Inference Period Close: **4 hours after the actual release timestamp** (typically around 18:00&nbsp;UTC).

Each day, an inference dataset is released containing the universe of tokens for which predictions are required. The inference data contains features but has no targets as they do not exist at the time of your submission. Your predictions will be scored against resolving targets from real market data in the future.

!!! note "Polling and parameter options"
    **Current vs Latest**: Use `"current"` to download today's active inference period (for making submissions). Use `"latest"` to download the most recently published period (which may be from a previous day if today's isn't ready yet).
    
    **Polling**: The inference file may not exist the very instant the clock strikes 14:00&nbsp;UTC. If you use `release_date="current"` with default parameters, polling is handled automatically. For `release_date="latest"`, no polling is needed since it always fetches the most recent available period.
    
    For manual polling, use `client.wait_for_inference_data("inference_data.parquet")` or gently poll the API until `GET /current` stops returning `404`. 

To download inference data:
```python
# Download the current active inference period (for today's submissions)
client.download_inference_data("current") # THIS WILL FAIL IF THERE IS NOT AN OPEN INFERENCE PERIOD

# Download the most recently available inference period
client.download_inference_data("latest")

# Download a specific date's inference data
client.download_inference_data("2024-12-15")
```

| id      | eodhd_id              | feature_1_lag15 | feature_1_lag10 | feature_1_lag5 | feature_1_lag0 | ... |
|---------|----------------------|----------------|----------------|-----------------|-----------------|-----|
| BABY    | BABY32198-USD.CC     | 0.123          | 0.145          | 0.112           | 0.156           | ... |
| OM      | OM-USD.CC            | 0.456          | 0.423          | 0.467           | 0.401           | ... |
| IOTA    | IOTA-USD.CC          | 0.789          | 0.812          | 0.798           | 0.823           | ... |
| MOODENG | MOODENG33093-USD.CC  | 0.234          | 0.267          | 0.223           | 0.289           | ... |
| ENS     | ENS-USD.CC           | 0.567          | 0.534          | 0.578           | 0.512           | ... |

!!! tip
    You do *not* need to use the features included in the inference data. You can use and are encouraged to use your own features. It may still be helpful to use our inference data and id mappings to use the same universe of assets.

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


### Metrics
1) [Symmetric Normalized Discounted Cumulative Gain (NDCG@40)](https://docs.crowdcent.com/scoring/#symmetric-normalized-discounted-cumulative-gain-symmetric-NDCGk)


When you see NDCG@40, think: "how well did I rank the top 40 assets and how well did I rank the bottom 40 assets?" With ~170 tokens in the universe, k=40 represents approximately the top/bottom 20-25% of assets. This metric equally rewards both:

  - **Top 40 identification**: Finding the tokens that will have the highest returns (for long positions)
  - **Bottom 40 identification**: Finding the tokens that will have the lowest returns (for short positions or avoidance)
  
  The logarithmic discount means getting the #1 ranked token correct is much more valuable than getting the #40 ranked token correct. A perfect NDCG@40 score of 1.0 means you perfectly ranked both tails of the distribution. This metric is particularly valuable for portfolio construction where you want to maximize exposure to the best performers while avoiding or shorting the worst.

!!! note "Random Baseline"
      Random predictions score approximately 0.55 for NDCG@40 with ~170 tokens, not 0.5. See the [detailed explanation](https://docs.crowdcent.com/scoring/#interpretation) for why this happens.

2) [Spearman Correlation](https://docs.crowdcent.com/scoring/#spearman-correlation)

Spearman's rank correlation (ρ) measures how well your predicted ranks align with the true ranks across the entire universe of ~170 tokens. Unlike NDCG@40 which focuses on the 40 extremes, ρ treats all rank positions in the entire universe equally.

### Composite Percentile

During the initial *warm-up* phase of the challenge, the goal is to maximize all metrics across all timeframes equally. Since NDCG@40 (0-1 range) and Spearman correlation (-1 to 1 range) have different scales and distributions, we use a **composite percentile** for fair comparison.

The **composite percentile** is calculated as the average of your percentile rankings across all four metrics:

$$
\text{Composite Percentile} = \frac{1}{4} \times \left(
\begin{array}{l}
\text{percentile(NDCG@40}_{10d}) + \\
\text{percentile(NDCG@40}_{30d}) + \\
\text{percentile(spearman}_{10d}) + \\
\text{percentile(spearman}_{30d})
\end{array}
\right)
$$

Where each percentile represents your ranking (0-100) compared to other participants for that specific metric for a given day/inference period.

**Important:** Composite percentiles are only calculated when **ten (10) or more valid submissions** (counted by submission slots, not users) are received for a given day. If fewer than ten submissions are present, the composite percentile will not be calculated, and you'll need to look at absolute metric scores instead.

As we learn more about the challenge's metamodel, we may adjust the weighting or add/remove metrics.

### Score Ranges and Percentile Rankings

**Raw Score Ranges:**

- **NDCG@40**: Ranges from 0.0 (worst possible) to 1.0 (perfect ranking of top and bottom 40 assets)
- **ρ (Spearman's Rank Correlation)**: Ranges from -1.0 (perfect inverse ranking) to 1.0 (perfect ranking), with 0.0 indicating random performance

**Why Are Scores Typically Low?**

Financial markets are characterized by extremely high noise-to-signal ratios. Seemingly "low" scores can be quite competitive in this domain. Additionally, market regimes shift over time, causing the distribution of achievable scores to fluctuate significantly.

**Percentile Rankings: Your Most Reliable Metric**

CrowdCent calculates **percentile rankings** that show where you stand relative to other participants. These percentiles are recalculated daily.

Tracking your percentile scores over time is often more informative than focusing on absolute scores, as it accounts for evolving competition and regime shifts that affect all participants. A model that consistently ranks in the 75th percentile across different market conditions can often be more valuable than one that occasionally achieves top scores but performs poorly in other regimes.

!!! warning "Minimum submissions for percentile"
    Percentiles only calculated when **ten (10) or more valid submissions** are received for a given day. If fewer than ten submissions are present, you'll see individual metric scores, but no percentile.

## Meta Model

The CrowdCent Meta Model aggregates predictions from all participants, representing a "wisdom of the crowd" which is made available to all users with a valid CrowdCent account. This may change in the future with no notice.

!!! warning "Meta Model Disclaimer"
    Meta model signals are released for informational and educational purposes only. Not financial, investment, or trading advice. CrowdCent disclaims all liability for any losses, damages, or consequences arising from use of the meta model. Users assume all risks.

### Construction Methodology

The meta model is currently constructed daily using a **simple, naive average** of all submission slots:

1. **Uniform Ranking**: Each individual submission's predictions are first converted to uniform rankings [0, 1] for each prediction column (`pred_10d`, `pred_30d`)
2. **Missing ID Handling**: Any asset IDs missing from individual submissions are filled with neutral rankings of 0.5 *after* the uniform ranking step
3. **Averaging**: The meta model takes the arithmetic mean of all normalized rankings across all valid submissions for each day

!!! note "Future Enhancements"
    The current simple averaging approach is intentionally naive but effective as a starting point. Future versions may incorporate more sophisticated weighting schemes based on historical performance, tail weighting, or other ensemble methods.

### Access and Downloads

The meta model is available through multiple channels:

- Via web: [https://crowdcent.com/challenge/hyperliquid-ranking/meta-model/](https://crowdcent.com/challenge/hyperliquid-ranking/meta-model/)
- Via API: `client.download_meta_model(dest_path="meta_model.parquet")`

The meta model is a parquet file with the following columns. New predictions are added daily, creating a time series with multiple release dates as shown in this sample:

| id      | pred_10d | pred_30d | release_date |
|---------|----------|----------|--------------|
| BTC     | 0.85     | 0.82     | 2024-12-15   |
| ETH     | 0.74     | 0.78     | 2024-12-15   |
| SOL     | 0.91     | 0.89     | 2024-12-15   |
| BTC     | 0.85     | 0.82     | 2024-12-16   |
| ETH     | 0.74     | 0.78     | 2024-12-16   |
| SOL     | 0.91     | 0.83     | 2024-12-16   |
| ...     | ...      | ...      | ...          |