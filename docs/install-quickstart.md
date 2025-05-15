## Install the client
=== "Using uv (Recommended)"

    ```bash
    uv add crowdcent-challenge
    ```

=== "Using pip"

    ```bash
    pip install crowdcent-challenge
    ```

## Get an API Key

You need an API key to use the CrowdCent Challenge API. You can get your key by clicking "Generate New Key" on your [profile page](https://crowdcent.com/profile). Write it down, as you won't be able to access it after you leave the page.

[![API keys](overrides/assets/images/api_keys.png)](https://crowdcent.com/profile){:target="_blank"}

## Authenticate and Initialize the Client

The API client requires authentication using your API key. This can be provided directly or via environment variables. You can interact with the API using the Python client or the CLI.

=== "Python"

    ```python
    from crowdcent_challenge import ChallengeClient, CrowdCentAPIError

    challenge_slug = "hyperliquid-ranking"  # Replace with your challenge
    api_key = "your_api_key_here" # Replace with your actual key
    client = ChallengeClient(challenge_slug=challenge_slug, api_key=api_key)
    ```

=== "CLI"
   
    ```bash
    export CROWDCENT_API_KEY=your_api_key_here # Set the environment variable
    echo "CROWDCENT_API_KEY=your_api_key_here" > .env # or create .env

    # Set the default challenge
    crowdcent set-default-challenge hyperliquid-ranking

    # Check current default challenge
    crowdcent get-default-challenge
    ```
    !!! note
        With a default challenge set, you can run most commands without explicitly specifying the challenge. If you need to override the default for a specific command, use the `--challenge` or `-c` option.

## Working with Challenges

Get details for a challenge or switch between different challenges.

=== "Python"

    ```python
    # Get details for the current challenge
    challenge = client.get_challenge()
    print(f"Challenge: {challenge['name']}")
    print(f"Description: {challenge['description']}")

    # Switch to a different challenge
    new_challenge_slug = "another-challenge"  # Replace with another actual challenge slug
    client.switch_challenge(new_challenge_slug)

    # Now all operations will be for the new challenge
    new_challenge = client.get_challenge()
    print(f"Switched to: {new_challenge['name']}")
    ```

=== "CLI"

    ```bash
    # List all available challenges
    crowdcent list-challenges

    # Get details for the default challenge
    crowdcent get-challenge
    
    # Or specify a challenge
    crowdcent get-challenge --challenge hyperliquid-ranking
    ```

## Working with Training Data

Access training datasets for a challenge, including listing available versions, getting the latest version, and downloading datasets.

=== "Python"

    ```python
    # List all training datasets for the current challenge
    training_datasets = client.list_training_datasets()
    for dataset in training_datasets:
        print(f"Version: {dataset['version']}, Is Latest: {dataset['is_latest']}")

    # Get the latest training dataset
    latest_dataset = client.get_latest_training_dataset()
    print(f"Latest Version: {latest_dataset['version']}")
    print(f"Download URL: {latest_dataset['download_url']}")

    # Download a training dataset file
    version = "1.0"  # or "latest" for the latest version
    output_path = "data/training_data.parquet"
    client.download_training_dataset(version, output_path)
    print(f"Dataset downloaded to {output_path}")
    ```

=== "CLI"

    ```bash
    # List all training datasets
    crowdcent list-training-data

    # Get details about the latest training dataset
    crowdcent get-latest-training-data

    # Get details about a specific training dataset version
    crowdcent get-training-data 1.0

    # Download latest version
    crowdcent download-training-data latest -o ./data/training_data.parquet
    
    # Or a specific version
    crowdcent download-training-data 1.0 -o ./data/training_data.parquet
    ```

## Working with Inference Data

Manage inference data periods, including listing available periods, getting details for the current period, and downloading inference features.

=== "Python"

    ```python
    # List all inference data periods for the current challenge
    inference_periods = client.list_inference_data()
    for period in inference_periods:
        print(f"Release Date: {period['release_date']}, Deadline: {period['submission_deadline']}")

    # Get the current inference period
    current_period = client.get_current_inference_data()
    print(f"Current Period Release Date: {current_period['release_date']}")
    print(f"Submission Deadline: {current_period['submission_deadline']}")
    print(f"Time Remaining: {current_period['time_remaining']}")

    # Download inference features
    release_date = "2025-01-15"  # or "current" for the current period
    output_path = "data/inference_features.parquet"
    client.download_inference_data(release_date, output_path)
    print(f"Inference data downloaded to {output_path}")
    ```

=== "CLI"

    ```bash
    # List all inference data periods
    crowdcent list-inference-data

    # Get details about the current inference period
    crowdcent get-current-inference-data

    # Get details about a specific inference period
    crowdcent get-inference-data 2025-01-15

    # Download current inference data
    crowdcent download-inference-data current -o ./data/inference_features.parquet
    
    # Or specific date
    crowdcent download-inference-data 2025-01-15 -o ./data/inference_features.parquet
    ```

## Working with the Meta Model

Download the consolidated meta model for a challenge. The meta model typically represents an aggregation of all valid user submissions for past inference periods.

=== "Python"

    ```python
    output_path = "data/meta_model.parquet"
    client.download_meta_model(output_path)
    print(f"Meta model downloaded to {output_path}")
    ```

=== "CLI"

    ```bash
    crowdcent download-meta-model -o ./data/meta_model.parquet
    ```

## Submitting Predictions

Submit your model's predictions for the current inference period. The file must include an `id` column and the specific prediction columns required by the challenge (e.g., `pred_10d`, `pred_30d` for some challenges, or `pred_1M`, `pred_3M`, etc., for others). Always check the specific challenge documentation for the exact column names.

=== "Python"

    ```python
    import polars as pl
    import numpy as np

    # Create or load your predictions
    inference_data = pl.read_parquet("inference_data.parquet")
    predictions = inference_data.with_columns([
        pl.Series("pred_10d", np.random.random(len(inference_data))).cast(pl.Float64),
        pl.Series("pred_30d", np.random.random(len(inference_data))).cast(pl.Float64),
    ])

    # Save predictions to a Parquet file
    predictions_file = "my_predictions.parquet"
    predictions.write_parquet(predictions_file)

    # Submit to the current challenge
    submission = client.submit_predictions(predictions_file)
    print(f"Submission successful! ID: {submission['id']}")
    print(f"Status: {submission['status']}")

    # You can specify a submission slot (1-5)
    submission = client.submit_predictions(predictions_file, slot=2)
    ```

=== "CLI"

    ```bash
    # Submit predictions to the default challenge
    crowdcent submit my_predictions.parquet
    
    # Submit to a specific slot
    crowdcent submit my_predictions.parquet --slot 2
    
    # Submit to a specific challenge (overriding default)
    crowdcent submit my_predictions.parquet --challenge hyperliquid-ranking
    ```

## Retrieving Submissions

Manage and review your submissions for a challenge, including listing all submissions, filtering by period, and getting details for a specific submission.

=== "Python"

    ```python
    # List your submissions for the current challenge
    submissions = client.list_submissions()
    for submission in submissions:
        print(f"Submission ID: {submission['id']}, Status: {submission['status']}")

    # Filter submissions by period
    # Get submissions for the current period only
    current_submissions = client.list_submissions(period="current")

    # Or for a specific period
    date_submissions = client.list_submissions(period="2025-01-15")

    # Get details for a specific submission
    submission_id = 123  # Replace with actual submission ID
    submission = client.get_submission(submission_id)
    print(f"Submitted at: {submission['submitted_at']}")
    print(f"Status: {submission['status']}")
    if submission['score_details']:
        print(f"Score Details: {submission['score_details']}")
    ```

=== "CLI"

    ```bash
    # List all submissions
    crowdcent list-submissions
    
    # Filter by period
    crowdcent list-submissions --period current
    crowdcent list-submissions --period 2025-01-15

    # Get details about a specific submission
    crowdcent get-submission 123
    ```

## Listing Available Challenges

Before working with a specific challenge, you may want to list all available challenges.

=== "Python"

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

=== "CLI"

    ```bash
    crowdcent list-challenges
    ```

## Working with Multiple Challenges

If you need to work with multiple challenges simultaneously, you can use either multiple client instances or override the default challenge.

=== "Python"

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

=== "CLI"

    ```bash
    # Set default challenge
    crowdcent set-default-challenge challenge-a
    
    # Work with default challenge
    crowdcent get-latest-training-data
    
    # Override for a specific command
    crowdcent get-latest-training-data --challenge challenge-b
    ```