## Install the client
=== "Using uv (Recommended)"

    ```bash
    uv pip install crowdcent-challenge
    ```

=== "Using pip"

    ```bash
    pip install crowdcent-challenge
    ```

## Get an API Key

You need an API key to use the CrowdCent Challenge API. You can get your key by clicking "Generate New Key" on your [profile page](https://crowdcent.com/profile). Write it down, as you won't be able to access it after you leave the page.

[![API keys](overrides/assets/images/api_keys.png)](https://crowdcent.com/profile){:target="_blank"}

## Initialize the ChallengeClient

The primary way to interact with the API is through the `ChallengeClient`, which is designed to work with a specific challenge. If you prefer to use the CLI, you can refer to the [CLI documentation](cli.md). Initialize the client for a specific challenge by providing the challenge slug and your API key.

```python
from crowdcent_challenge import ChallengeClient, CrowdCentAPIError

challenge_slug = "crypto-ranking"  # Replace with the challenge slug you want to work on
api_key = "your_api_key_here" # Replace with your actual key
client = ChallengeClient(challenge_slug=challenge_slug, api_key=api_key)
```

!!! note
    You can alternatively set the `CROWDCENT_API_KEY` environment variable or create a `.env` file in your project root:
    ```
    CROWDCENT_API_KEY=your_api_key_here
    ```
    If the API key is not provided and cannot be found in the environment or `.env` file, an `AuthenticationError` will be raised.

## Working with a Challenge

Get details for the current challenge:

```python
challenge = client.get_challenge()
print(f"Challenge: {challenge['name']}")
print(f"Description: {challenge['description']}")
```

If you want to switch to a different challenge with the same client instance:

```python
new_challenge_slug = "another-challenge"  # Replace with another actual challenge slug
client.switch_challenge(new_challenge_slug)

# Now all operations will be for the new challenge
new_challenge = client.get_challenge()
print(f"Switched to: {new_challenge['name']}")
```

## Working with Training Data

List all training datasets for the current challenge:

```python
training_datasets = client.list_training_datasets()
for dataset in training_datasets:
    print(f"Version: {dataset['version']}, Is Latest: {dataset['is_latest']}")
```

Get the latest training dataset:

```python
latest_dataset = client.get_latest_training_dataset()
print(f"Latest Version: {latest_dataset['version']}")
print(f"Download URL: {latest_dataset['download_url']}")
```

Download a training dataset file:

```python
version = "1.0"  # or "latest" for the latest version
output_path = "data/training_data.parquet"
client.download_training_dataset(version, output_path)
print(f"Dataset downloaded to {output_path}")
```

## Working with Inference Data

List all inference data periods for the current challenge:

```python
inference_periods = client.list_inference_data()
for period in inference_periods:
    print(f"Release Date: {period['release_date']}, Deadline: {period['submission_deadline']}")
```

Get the current inference period:

```python
current_period = client.get_current_inference_data()
print(f"Current Period Release Date: {current_period['release_date']}")
print(f"Submission Deadline: {current_period['submission_deadline']}")
print(f"Time Remaining: {current_period['time_remaining']}")
```

Download inference features:

```python
release_date = "2025-01-15"  # or "current" for the current period
output_path = "data/inference_features.parquet"
client.download_inference_data(release_date, output_path)
print(f"Inference data downloaded to {output_path}")
```

## Working with the Meta Model

Download the consolidated meta model for the current challenge:

```python
output_path = "data/meta_model.parquet"
client.download_meta_model(output_path)
print(f"Meta model downloaded to {output_path}")

## Submitting Predictions

Submit predictions for the current inference period:

```python
import polars as pl

# Create or load your predictions
# The file must include columns: id, pred_1M, pred_3M, pred_6M, pred_9M, pred_12M
predictions = pl.DataFrame({
    "id": [1, 2, 3],
    "pred_1M": [0.5, -0.3, 0.1],
    "pred_3M": [0.7, -0.2, 0.2],
    "pred_6M": [0.8, -0.1, 0.3],
    "pred_9M": [0.9, 0.0, 0.4],
    "pred_12M": [1.0, 0.1, 0.5]
})

# Save predictions to a Parquet file
predictions_file = "my_predictions.parquet"
predictions.write_parquet(predictions_file)

# Submit to the current challenge
submission = client.submit_predictions(predictions_file)
print(f"Submission successful! ID: {submission['id']}")
print(f"Status: {submission['status']}")

```

## Retrieving Submissions

List your submissions for the current challenge:

```python
submissions = client.list_submissions()
for submission in submissions:
    print(f"Submission ID: {submission['id']}, Status: {submission['status']}")
```

You can filter submissions by period:

```python
# Get submissions for the current period only
current_submissions = client.list_submissions(period="current")

# Or for a specific period
date_submissions = client.list_submissions(period="2025-01-15")
```

Get details for a specific submission:

```python
submission_id = 123  # Replace with actual submission ID
submission = client.get_submission(submission_id)
print(f"Submitted at: {submission['submitted_at']}")
print(f"Status: {submission['status']}")
if submission['score_details']:
    print(f"Score Details: {submission['score_details']}")
```

## Listing Available Challenges

Before initializing a client for a specific challenge, you may want to list all available challenges:

```python
from crowdcent_challenge import ChallengeClient, CrowdCentAPIError

# List all challenges using the class method
try:
    challenges = ChallengeClient.list_all_challenges()
    for challenge in challenges:
        print(f"Challenge: {challenge['name']} (Slug: {challenge['slug']})")
except CrowdCentAPIError as e:
    print(f"Error listing challenges: {e}")
```

## Working with Multiple Challenges

If you need to work with multiple challenges simultaneously, create separate clients:

```python
# Initialize clients for different challenges
client_a = ChallengeClient(challenge_slug="challenge-a")
client_b = ChallengeClient(challenge_slug="challenge-b")

# Use each client for its respective challenge
dataset_a = client_a.get_latest_training_dataset()
dataset_b = client_b.get_latest_training_dataset()

print(f"Challenge A latest dataset: {dataset_a['version']}")
print(f"Challenge B latest dataset: {dataset_b['version']}")
```