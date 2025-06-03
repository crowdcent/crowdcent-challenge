## Objective
The Hyperliquid Ranking Challenge requires participants to rank crypto assets on the Hyperliquid decentralized derivatives exchange by their expected relative returns over the next 10 and 30 days. The challenge universe comprises approximately 180 liquid tokens on the Hyperliquid protocol. This universe may change periodically, with tokens added or removed to ensure it remains as actionable as possible. If a token does not have enough volume or liquidity, it will likely be removed from the universe.

## Data
### Inference data
- Inference Period Open: Approximately 14:00 UTC
- Inference Period Close: Approximately 18:00 UTC

Each day, an inference dataset is released containing the universe of tokens for which predictions are required. The tickers are generally the same as you would find in the training data, but new additions or removals are possible. The inference data contains features but has no targets as they do not exist at the time of your submission. Your predictions will be scored against resolving targets from real market data in the future.

To download inference data for the current period:
```python
from crowdcent_challenge.client import CrowdCentClient

client = CrowdCentClient(challenge_slug="hyperliquid-ranking")
client.download_inference_data()
```

| id      | eodhd_id              | feature_1, feature_2, ... |
|---------|----------------------|---------------------------|
| BABY    | BABY32198-USD.CC     | 0.1, 0.2, 0.3             |
| OM      | OM-USD.CC            | 0.4, 0.5, 0.6             |
| IOTA    | IOTA-USD.CC          | 0.7, 0.8, 0.9             |
| MOODENG | MOODENG33093-USD.CC  | 1.0, 1.1, 1.2             |
| ENS     | ENS-USD.CC           | 1.3, 1.4, 1.5             |

We currently provide id (the hyperliquid id) and eodhd_id (the eodhd id) for each asset. You can request for additional id mappings from other data vendors to be added to the inference data if data licenses allow.

If you're using CrowdCent's training data to build your models, the inference data version and its features will always match that of the *latest* training data version.

### Training data
You can download our training data from [crowdcent.com/challenge/hyperliquid-ranking](https://crowdcent.com/challenge/hyperliquid-ranking) or via the CrowdCent client:
```python
from crowdcent_challenge.client import CrowdCentClient

client = CrowdCentClient(challenge_slug="hyperliquid-ranking")
client.download_training_data()
```

| id      | eodhd_id              | feature_1, feature_2, ... | target_10d, target_30d |
|---------|----------------------|---------------------------|------------------------|
| BABY    | BABY32198-USD.CC     | 0.1, 0.2, 0.3             | 0.15, 0.25             |
| OM      | OM-USD.CC            | 0.4, 0.5, 0.6             | 0.35, 0.45             |
| IOTA    | IOTA-USD.CC          | 0.7, 0.8, 0.9             | 0.55, 0.65             |
| MOODENG | MOODENG33093-USD.CC  | 1.0, 1.1, 1.2             | 0.75, 0.85             |
| ENS     | ENS-USD.CC           | 1.3, 1.4, 1.5             | 0.95, 1.05             |

You can also make your own data using sources like ccxt, coingecko, coinmarketcap, eodhd, or yfinance.

Targets are the rankings of an asset's 10d and 30d forward relative returns (with a 1d lag) and Daily Close Price Time is: 24:00 UTC 

#### Target Timing and Trading Implications

**Why the 1-day lag?**
At 14:00 UTC when predictions are made, only data through the previous day's 24:00 UTC close is available. The lag ensures predictions use only historical data while forecasting returns starting from tonight's close.

**Timeline:**
```
Day D-1: 24:00 UTC → Close price finalized (latest available data)
Day D:   14:00 UTC → Inference data released
         14:00-18:00 UTC → Inference period
         24:00 UTC → Prediction period starts (Day D close)
Day D+10/30: 24:00 UTC → Prediction period ends
```

- Predictions rank assets by expected performance over the next 10/30 days starting from tonight's close
- Rankings are relative. They say nothing about the expected absolute performance of an asset.

## Submitting predictions
Minimum of 80 ids from the inference data are required for a valid submission. 

- `id`: The id of the asset on Hyperliquid.
- `pred_10d`: A float between 0 and 1 representing the predicted rank for the 10-day horizon.
- `pred_30d`: A float between 0 and 1 representing the predicted rank for the 30-day horizon.


To submit predictions:
```python
client = CrowdCentClient(challenge_slug="hyperliquid-ranking")
client.submit_predictions(df=predictions_df)
```


| id      | pred_10d | pred_30d |
|---------|----------|----------|
| BABY    | 0.2      | 0.3      |
| OM      | 0.4      | 0.5      |
| IOTA    | 0.1      | 0.2      |
| MOODENG | 1.0      | 1.0      |
| ENS     | 0.3      | 0.4      |

!!! Note
    All data and predictions must be in parquet format.

## Scoring and Evaluation
Before scoring, for each prediction timeframe, ids are uniform ranked [0, 1], and any missing ids are filled with 0.5.

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

Because absolute scores vary with market conditions, CrowdCent calculates **percentile rankings** that show where you stand relative to other participants. These percentiles are recalculated daily.

Tracking your percentile rank over time is often more informative than focusing on absolute scores, as it accounts for:

- Changing market difficulty
- Evolving competition
- Regime shifts that affect all participants

!!! tip "Focus on Consistency"
    A model that consistently ranks in the 75th percentile across different market conditions is often more valuable than one that occasionally achieves top scores but performs poorly in other regimes.