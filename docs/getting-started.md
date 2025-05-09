## Register for a CrowdCent account
Sign up for a CrowdCent account [here](https://crowdcent.com/accounts/signup/) or sign in with your GitHub account. We require an email verification step to ensure the account is real.

## Explore Challenges
Once logged in, you'll land on the [Challenge List](https://crowdcent.com/challenge) page. Browse through the available challenges to find one that interests you. Each challenge card will give you a brief overview. Click on a challenge to see more details.

![Challenge List](overrides/assets/images/challenge_list.png)

## Download Data
On the detail page for your chosen challenge (e.g. [hyperliquid-ranking](https://crowdcent.com/challenge/hyperliquid-ranking)), you will find:

- A section to download the latest **Training Data**. You'll need this to train your model.
- Information about the current or most recent **Inference Data** period. If a period is active, you can download the inference features here.

## Build a Model
Using the downloaded training data, build a model to predict the challenge target(s). You can refer to our tutorial notebooks (if available in the challenge description or docs) for examples.

## Understand Inference Periods & Submission Slots
- The Challenge Detail page will display information about the current **Inference Data** period, including its release date and submission deadline.
- You have a specific number of **submission slots** (e.g., up to 5) for each inference period. You can choose which slot to use for each submission.

[![Submission Panel](overrides/assets/images/inference_period.png)](https://crowdcent.com/challenge){:target="_blank"}
*The submission panel on the Challenge Detail page shows active inference periods and your submission slots.*

## Submit Predictions

There are two main ways to submit your predictions:

### 1. Via the Website (UI)
- During an active inference period, go to the Challenge Detail page.
- In the submission panel, select an available **slot**.
- Upload your prediction file (typically a Parquet file).
- If you upload a file to a slot that already has a prediction, it will be **overwritten**.

### 2. Programmatically (via API)
- Go to your **User Profile** page (accessible from the top navigation bar when logged in).
- In the "API Keys" section, you can generate a new API key. Give it a descriptive name. **Store this key securely as it will not be shown again.**
- Use this API key with the `crowdcent-challenge` Python package to submit your predictions. See our [Python client quickstart guide](install-quickstart.md) for more details.

## Wait for Scores
After an inference period's submission deadline passes, predictions will be evaluated. Your submission status and scores will be updated on your profile and the challenge leaderboard.

For more details on how scores are calculated and what the scores mean, see the [Scoring](scoring.md) page.

## Check the Leaderboard
Navigate to the [Leaderboard](https://crowdcent.com/leaderboard) page to see how your submissions rank against other participants for each challenge. You can switch the leaderboard by challenge, sort by different scores, and view results by user or by individual submission slots.

![Leaderboard](overrides/assets/images/leaderboard.png)

