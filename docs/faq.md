# Frequently Asked Questions (FAQ)

## General & Getting Started

### What is the CrowdCent Challenge?

The CrowdCent Challenge is a series of open data science competitions (challenges) focused on predicting investment/market outcomes. Participants use various datasets to build machine learning models that predict future returns over various time horizons. Submissions are used to create meta-models that can be turned into investable portfolios.

### How do I get started?

Refer to the [Installation & Quick Start Guide](install-quickstart.md) for detailed steps.

1.  **Install the client library:** We recommend using uv: `uv pip install crowdcent-challenge`. Alternatively, use `pip install crowdcent-challenge`.
2.  **Get an API Key:** Visit your [profile settings](https://crowdcent.com/profile/settings/) and create a new key. Save it securely.
3.  **Set up Authentication:** Provide your API key either when initializing the Python `ChallengeClient`, setting the `CROWDCENT_API_KEY` environment variable, or placing it in a `.env` file (`CROWDCENT_API_KEY=your_key_here`) in your project directory.
4.  **Explore:** Use the Python client or the `crowdcent` CLI to list available challenges (`crowdcent list-challenges`).
5.  **Download Data:** Choose a challenge and download the training and inference data using the client or CLI (e.g., `crowdcent download-training-data <challenge_slug> latest`).
6.  **Build & Submit:** Train your model and submit your predictions in the required format.



### Who can participate?

The challenge is open to anyone interested in data science, machine learning, and finance. Check the [terms of service](https://crowdcent.com/terms/) for more details.

## API Key & Authentication

### How do I get an API Key?

Go to your [profile settings](https://crowdcent.com/profile/settings/) after logging in. In the **API Keys** section, click **New Key**, enter a name, and click **Create**.

### How do I use my API Key?

The `crowdcent-challenge` library (both Python client and CLI) automatically looks for your API key in the following order:

1.  Passed directly to the `ChallengeClient` initializer (`api_key=...`).
2.  The `CROWDCENT_API_KEY` environment variable.
3.  A `.env` file in your current working directory containing `CROWDCENT_API_KEY=your_key_here`.

### What if my API key doesn't work?

Ensure you copied the key correctly and included the `ApiKey ` prefix if using tools like Swagger UI directly (the client library handles this automatically). Verify it hasn't been revoked in your [profile settings](https://crowdcent.com/profile/settings/). If issues persist, generate a new key or contact support.

## Python Client vs. CLI

### What's the difference between the `ChallengeClient` (Python) and the `crowdcent` CLI?

They both interact with the same CrowdCent API.

*   **`ChallengeClient` (Python):** Designed for programmatic use within your modeling scripts or notebooks. Ideal for automating data downloads, processing, and prediction uploads.
*   **`crowdcent` (CLI):** A command-line tool for manual operations like listing challenges, downloading specific data files, checking submission status, etc., directly from your terminal.

### Which one should I use?

Use the `ChallengeClient` within your Python code for automation and integration with your modeling workflow. Use the `crowdcent` CLI for quick checks, manual downloads, or exploring the available challenges and data without writing Python code.

## Data

### What format is the data provided in?

All datasets (training data, inference features, meta-models) and submission files are in the Apache Parquet (`.parquet`) format. This columnar format is efficient for the type of data used in the challenge.

### What kind of features are included?



*   **Important Note:** Some challenges or training datasets might only provide target labels and identifiers (`id`, `date`, `target_10d`, `target_30d`). In these cases, participants are expected to source or engineer their own relevant features.
*   Features are often renamed to simple names like `feature_1`, `feature_2`, etc. Refer to the specific challenge rules for details on the data provided for each competition.

### What am I predicting?

Your goal is to to predict relative performance metrics (e.g., returns) for different future time horizons based on the provided features. The required prediction columns vary by challenge, but may look like `pred_10d`, `pred_30d`. Always check the specific challenge rules for the exact requirements.

### What's the difference between Training and Inference data?

*   **Training Data:** Used to train your models. Contains historical data, including target variables (e.g., `target_10d`, `target_30d`, ...) and identifiers (`id`, dates). It *may* also contain pre-computed features, but sometimes you will need to generate your own features based on the provided IDs and timestamps. Training datasets are versioned (e.g. v1.0, v1.1, etc.).
*   **Inference Data:** Contains features (if provided by the challenge) and identifiers (`id`) for a *new* period but *without* the target labels. This is the data you use (along with any features you generate) to make predictions for submission. Inference data is released periodically.

### What is the 'Meta-Model'?

The meta-model typically represents an aggregation (e.g., an average or ensemble) of all valid user submissions for past inference periods within a specific challenge. It can serve as a benchmark or potentially as an additional feature for your own models. You can download it via the client or CLI, or run it through the [Simulator](simulator.md) as a long/short portfolio without writing any code. Real-time predictions are available to **Challenger+** tier (100+ CC Points); all other users receive 90-day delayed data. Performance scores are open to everyone. See [CC Points](points-system.md) for tier details.

### Can I trade the meta-model?

You can simulate it today: the [Simulator](simulator.md) backtests the meta-model as a long/short Hyperliquid portfolio with configurable weighting, risk, and cadence controls, and higher [CC Point](points-system.md) tiers unlock more of them. Live trading through CrowdCent is in a staff preview. For self-custody execution from your own machine, there is [cc-liquid](https://github.com/crowdcent/cc-liquid), our open-source CLI rebalancer. Nothing here is investment advice; see the [disclaimer](disclaimer.md).

## Submissions

### What format does my submission file need to be?

Your submission must be a Parquet file containing an `id` column that matches the IDs from the corresponding inference dataset, and all the required prediction columns (e.g., `pred_1M`, `pred_3M`, etc.). All prediction columns must contain numeric values, and no missing values are allowed.

### How do I submit my predictions?

*   **Python:** Use `client.submit_predictions("path/to/your/predictions.parquet")`.
*   **CLI:** Use `crowdcent submit path/to/your/predictions.parquet`.

If a submission window is currently open, your prediction is submitted immediately. If no window is open, your prediction is **queued** and will be automatically submitted when the next window opens.

By default, submissions are also queued for the following period (auto-rollover). Use `queue_next=False` (Python) or `--no-queue-next` (CLI) to opt out. The Python client and CLI also accept `is_experimental` / `--experimental` and `notes` / `--notes` — see [What is an experimental submission?](#what-is-an-experimental-submission) below.

### How often can I submit?

You can submit multiple times for an active inference period. Your latest valid submission before the deadline is the one that counts for scoring. Check the specific challenge rules.

### How do I check my submission status?

*   **Python:** Use `client.list_submissions()` (optionally filter by `period='current'` or `period='YYYY-MM-DD'`) and `client.get_submission(<submission_id>)`.
*   **CLI:** Use `crowdcent list-submissions <challenge_slug>` (optionally filter with `--period current` or `--period YYYY-MM-DD`) and `crowdcent get-submission <challenge_slug> <submission_id>`.
Statuses include "pending", "processing", "evaluated" (or "scored"), and "error" (or "failed").

### What is an experimental submission?

A submission you mark as experimental gets scored against the non-experimental competitive field (so you can see your shadow percentile) but is excluded from the leaderboard, the meta-model, and your CC Points. It's the right tool for testing a new architecture or feature set without dragging your competitive standing.

### Why was my experimental submission rejected?

You need at least one **non-experimental** submission in another slot for the same period. This keeps the competitive pool honest and prevents experimental-only entries. Submit a non-experimental prediction to another slot first, or uncheck experimental on the current one.

### Do experimental submissions affect my CC Points or streak?

Experimental submissions are excluded from the [Performance Adjustment](points-system.md) (only your non-experimental slots feed your average percentile). They don't influence the meta-model's weighting either. Your non-experimental submission is what triggers daily base credit and feeds your streak — by design every period has at least one non-experimental sub when you also submit experimental, so streak/base-credit math is unchanged.

### Can other users see my experimental submissions or notes?

Experimental submissions are visible on profiles with an **experimental** label (triangular slot badge), and you can see other users' shadow percentiles. However, **notes are private to the submission owner**, i.e. only the note creator can see them.

### Can I edit notes after the period closes?

Yes. The notes endpoint is independent of the scoring lifecycle. Edit them inline on your own profile/scores pages at any time.

### Why did my live submission succeed but the queue copy was rejected?

The live save and the queue-for-next-period copy are processed independently — one can succeed while the other is rejected. If you submit experimental (`is_experimental=True`) without a non-experimental queued submission in another slot, the live save commits but the queue copy is rejected. Your live submission still counts. Either pass `queue_next=False`, or queue a non-experimental sub to another slot first.

## Scoring & Uniqueness

### What are uniqueness metrics?

Uniqueness metrics measure how differentiated your predictions are from the meta-model and whether that unique signal is actually predictive. See [Scoring - Uniqueness Metrics](scoring.md#uniqueness-metrics) for full details.

### Should I optimize for raw metrics or uniqueness metrics?

Both matter for [CC Points](points-system.md): raw metrics drive your [composite percentile](hyperliquid-ranking.md#composite-percentile) and uniqueness metrics drive your [unique composite percentile](hyperliquid-ranking.md#unique-composite-percentile), blended equally in the performance adjustment. A submission highly correlated with the meta adds little new information even if its raw scores are strong. The ideal is a model that scores well on raw metrics *and* provides differentiated signal. In practice, focusing on building a genuinely good model with your own data and features tends to naturally produce unique predictions.

### When did CC Points start using the uniqueness blend?

**July 1, 2026.** Inference periods released on or after that date use the 50/50 blend described in [Performance Adjustment](points-system.md#performance-adjustment-the-core-driver). Earlier periods keep the raw-only formula.

### What is the difference between corr_to_meta and unique_spearman?

`corr_to_meta` tells you *how similar* your predictions are to the meta-model (lower absolute value = more unique). `unique_spearman` tells you *whether your unique signal is predictive* -- it neutralizes out the meta component and then checks if the residual correlates with actual outcomes. You can be unique (low `corr_to_meta`) but wrong (negative `unique_spearman`), or you can be unique and right (low `corr_to_meta`, positive `unique_spearman`).

## Environment & Troubleshooting

### What version of Python should I use?

The client library requires Python 3.10 or higher (as specified in `pyproject.toml`).

### `uv` or `pip`?

We recommend using `uv` for faster dependency management (`uv pip install ...`). However, standard `pip install ...` also works.

### Where can I get help?

*   **Discord:** Join the [CrowdCent Discord server](https://discord.gg/v6ZSGuTbQS).
*   **Email:** Contact [info@crowdcent.com](mailto:info@crowdcent.com).

## Contributing

### How can I contribute?

Contributions to the `crowdcent-challenge` client library and its documentation are welcome! Please see the [Contributing Guidelines](contributing.md) for details on the standard GitHub workflow (fork, branch, commit, PR). 