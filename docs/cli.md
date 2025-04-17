# Command Line Interface (CLI) Usage

The package provides a command-line interface (CLI) called `crowdcent` for quick interactions with the API.

## Authentication

The CLI requires your API key for authentication. It looks for the key in the following order:

1.  The `CROWDCENT_API_KEY` environment variable.
2.  A `.env` file in the current working directory containing `CROWDCENT_API_KEY=your_key_here`.

If the key is not found, the CLI commands will fail with an authentication error.

## General Usage

```bash
crowdcent [OPTIONS] COMMAND [ARGS]...
```

Get help on the main command or subcommands:

```bash
crowdcent --help
crowdcent list-challenges --help
crowdcent submit --help
```

## Commands

### Challenges

* List active challenges:
    ```bash
    crowdcent list-challenges
    ```
    Output is JSON formatted.

* Get specific challenge details:
    ```bash
    crowdcent get-challenge <CHALLENGE_SLUG>
    ```
    Replace `<CHALLENGE_SLUG>` with the actual challenge slug (e.g., `crowdcent get-challenge main-challenge`). Output is JSON formatted.

### Training Data

* List training datasets for a challenge:
    ```bash
    crowdcent list-training-data <CHALLENGE_SLUG>
    ```
    Output is JSON formatted.

* Get the latest training dataset for a challenge:
    ```bash
    crowdcent get-latest-training-data <CHALLENGE_SLUG>
    ```
    Output is JSON formatted.

* Get specific training dataset details:
    ```bash
    crowdcent get-training-data <CHALLENGE_SLUG> <VERSION>
    ```
    Replace `<CHALLENGE_SLUG>` with the challenge slug and `<VERSION>` with the version string (e.g., `1.0`). Output is JSON formatted.

* Download training dataset file:
    ```bash
    crowdcent download-training-data <CHALLENGE_SLUG> <VERSION> [-o <OUTPUT_PATH>]
    ```
    -   Replace `<CHALLENGE_SLUG>` with the challenge slug.
    -   Replace `<VERSION>` with the version string or `latest` for the latest version.
    -   The `-o` or `--output` flag is optional. If omitted, the file is saved as `<CHALLENGE_SLUG>_training_v<VERSION>.parquet` in the current directory.
    
    Example:
    ```bash
    crowdcent download-training-data main-challenge 1.0 -o data/training_data.parquet
    ```

### Inference Data

* List inference data periods for a challenge:
    ```bash
    crowdcent list-inference-data <CHALLENGE_SLUG>
    ```
    Output is JSON formatted.

* Get the current inference data period for a challenge:
    ```bash
    crowdcent get-current-inference-data <CHALLENGE_SLUG>
    ```
    Output is JSON formatted.

* Get specific inference data period details:
    ```bash
    crowdcent get-inference-data <CHALLENGE_SLUG> <RELEASE_DATE>
    ```
    Replace `<CHALLENGE_SLUG>` with the challenge slug and `<RELEASE_DATE>` with the date in `YYYY-MM-DD` format. Output is JSON formatted.

* Download inference features file:
    ```bash
    crowdcent download-inference-data <CHALLENGE_SLUG> <RELEASE_DATE> [-o <OUTPUT_PATH>]
    ```
    -   Replace `<CHALLENGE_SLUG>` with the challenge slug.
    -   Replace `<RELEASE_DATE>` with the date in `YYYY-MM-DD` format or `current` for the current period.
    -   The `-o` or `--output` flag is optional. If omitted, the file is saved as `<CHALLENGE_SLUG>_inference_<RELEASE_DATE>.parquet` in the current directory.
    
    Example:
    ```bash
    crowdcent download-inference-data main-challenge 2025-01-15 -o data/inference_features.parquet
    ```

### Submissions

* List your submissions for a challenge:
    ```bash
    crowdcent list-submissions <CHALLENGE_SLUG> [--period <PERIOD>]
    ```
    -   Replace `<CHALLENGE_SLUG>` with the challenge slug.
    -   The `--period` flag is optional. It can be set to `current` for the current period or a date in `YYYY-MM-DD` format to filter by period.
    
    Output is JSON formatted.

* Get specific submission details:
    ```bash
    crowdcent get-submission <CHALLENGE_SLUG> <SUBMISSION_ID>
    ```
    Replace `<CHALLENGE_SLUG>` with the challenge slug and `<SUBMISSION_ID>` with the actual ID. Output is JSON formatted.

* Submit predictions:
    ```bash
    crowdcent submit <CHALLENGE_SLUG> <PATH_TO_PREDICTIONS_PARQUET>
    ```
    Replace `<CHALLENGE_SLUG>` with the challenge slug and `<PATH_TO_PREDICTIONS_PARQUET>` with the path to your prediction file.
    
    The file must be a valid Parquet file with the required columns: `id`, `pred_1M`, `pred_3M`, `pred_6M`, `pred_9M`, and `pred_12M`.
    
    Example:
    ```bash
    crowdcent submit main-challenge results/my_submission.parquet
    ```

## Error Handling

The CLI will print error messages to standard error if an API call fails or if invalid arguments are provided. 