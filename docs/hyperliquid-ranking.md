## Objective
The Hyperliquid Ranking Challenge requires participants to rank crypto assets on the Hyperliquid decentralized derivatives exchange by their expected relative returns over the next 10 and 30 days. The challenge universe comprises approximately 185 liquid tokens on the Hyperliquid protocol. This universe may change periodically, with tokens added or removed to ensure it remains as actionable as possible. If a token does not have enough volume or liquidity, it will likely be removed from the universe.

## Data
### Training data
You can download our obfuscated training data from [crowdcent.com/challenge/hyperliquid-ranking](https://crowdcent.com/challenge/hyperliquid-ranking) or via the CrowdCent client:
```python
from crowdcent_challenge.client import CrowdCentClient

client = CrowdCentClient()
client.download_training_data()
```

You can also make your own data using the following sources or more:

- cctx
- coingecko
- coinmarketcap
- eodhd
- yfinance

Daily Close Price Time: 24:00 UTC 
Targets are the rankings of an asset's 10d and 30d forward relative returns (with a 1d lag)


### Inference data
- Inference Period Open: Approximately 14:00 UTC
- Inference Period Close: Approximately 18:00 UTC

Each day, an inference dataset is released containing the universe of tokens for which predictions are required. The tickers are generally the same, but new additions or removals are possible.
To download inference data for the current period:
```python
from crowdcent_challenge.client import CrowdCentClient

client = CrowdCentClient()
client.download_inference_data()
```

If you're using CrowdCent's training data to build your models, the inference data version will always be the same as the *latest* training data version.


## Scoring and Evaluation
**Predictions:**
Minimum of 80 ids are required for a valid submission. Before scoring, ids are uniform ranked [0, 1], and missing ids are ranked 0.5.

- `pred_10d`: A float between 0 and 1 representing the predicted rank for the 10-day horizon.
- `pred_30d`: A float between 0 and 1 representing the predicted rank for the 30-day horizon.

!!! Note
    All data and predictions must be in parquet format.

**Metrics:**

- [Symmetric Normalized Discounted Cumulative Gain (NDCG@40)](https://docs.crowdcent.com/scoring/#symmetric-normalized-discounted-cumulative-gain-symmetric-ndcgk)
- [Spearman Correlation](https://docs.crowdcent.com/scoring/#spearman-correlation)