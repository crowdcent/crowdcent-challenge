# Client Library Quick Start

The primary way to interact with the API programmatically is through the `ChallengeClient`.

## Initialization

First, import and initialize the client. The API key is required for authentication.

```python
from crowdcent_challenge import ChallengeClient, CrowdcentAPIError

# Option 1: Pass the key directly
API_KEY = "your_api_key_here" # Replace with your actual key
client = ChallengeClient(api_key=API_KEY)

# Option 2: Set the CROWDCENT_API_KEY environment variable
# or create a .env file in your project root:
# CROWDCENT_API_KEY=your_api_key_here
# Then initialize without arguments:
# client = ChallengeClient()
```

!!! note
    If the API key is not provided and cannot be found in the environment or `.env` file, an `AuthenticationError` will be raised.

## Working with Challenges

List all active challenges:

```python
challenges = client.list_challenges()
for challenge in challenges:
    print(f"Challenge: {challenge['name']} (Slug: {challenge['slug']})")
```

Get details for a specific challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
challenge = client.get_challenge(challenge_slug)
print(f"Challenge: {challenge['name']}")
print(f"Description: {challenge['description']}")
```

## Working with Training Data

List all training datasets for a challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
training_datasets = client.list_training_datasets(challenge_slug)
for dataset in training_datasets:
    print(f"Version: {dataset['version']}, Is Latest: {dataset['is_latest']}")
```

Get the latest training dataset for a challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
latest_dataset = client.get_latest_training_dataset(challenge_slug)
print(f"Latest Version: {latest_dataset['version']}")
print(f"Download URL: {latest_dataset['download_url']}")
```

Download a training dataset file:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
version = "1.0"  # or "latest" for the latest version
output_path = "data/training_data.parquet"
client.download_training_dataset(challenge_slug, version, output_path)
print(f"Dataset downloaded to {output_path}")
```

## Working with Inference Data

List all inference data periods for a challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
inference_periods = client.list_inference_data(challenge_slug)
for period in inference_periods:
    print(f"Release Date: {period['release_date']}, Deadline: {period['submission_deadline']}")
```

Get the current inference period for a challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
try:
    current_period = client.get_current_inference_data(challenge_slug)
    print(f"Current Period Release Date: {current_period['release_date']}")
    print(f"Submission Deadline: {current_period['submission_deadline']}")
    print(f"Time Remaining: {current_period['time_remaining']}")
except CrowdcentAPIError as e:
    print(f"No active inference period found: {e}")
```

Download inference features:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
release_date = "2025-01-15"  # or "current" for the current period
output_path = "data/inference_features.parquet"
client.download_inference_data(challenge_slug, release_date, output_path)
print(f"Inference data downloaded to {output_path}")
```

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

# Submit to a specific challenge
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
try:
    submission = client.submit_predictions(challenge_slug, predictions_file)
    print(f"Submission successful! ID: {submission['id']}")
    print(f"Status: {submission['status']}")
except CrowdcentAPIError as e:
    print(f"Submission failed: {e}")
```

## Retrieving Submissions

List your submissions for a challenge:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
submissions = client.list_submissions(challenge_slug)
for submission in submissions:
    print(f"Submission ID: {submission['id']}, Status: {submission['status']}")
```

You can filter submissions by period:

```python
# Get submissions for the current period only
current_submissions = client.list_submissions(challenge_slug, period="current")

# Or for a specific period
date_submissions = client.list_submissions(challenge_slug, period="2025-01-15")
```

Get details for a specific submission:

```python
challenge_slug = "stock-prediction"  # Replace with actual challenge slug
submission_id = 123  # Replace with actual submission ID
submission = client.get_submission(challenge_slug, submission_id)
print(f"Submitted at: {submission['submitted_at']}")
print(f"Status: {submission['status']}")
if submission['score_details']:
    print(f"Score Details: {submission['score_details']}")
```