# CrowdCent CLI Reference

## Authentication

The CLI searches for your API key in this order:
1. `CROWDCENT_API_KEY` environment variable
2. `.env` file in the current directory with `CROWDCENT_API_KEY=your_key_here`

## Basic Usage

```bash
crowdcent [OPTIONS] COMMAND [ARGS]...
```

Show help:
```bash
crowdcent --help
crowdcent COMMAND --help
```

## Commands

### Challenge Management

#### list-challenges
```bash
crowdcent list-challenges
```
List all active challenges (JSON output).

#### get-challenge
```bash
crowdcent get-challenge <challenge_slug>
```
Get details for a specific challenge.

### Training Data

#### list-training-data
```bash
crowdcent list-training-data <challenge_slug>
```
List all available training datasets for the challenge.

#### get-latest-training-data
```bash
crowdcent get-latest-training-data <challenge_slug>
```
Get metadata for the most recent training dataset.

#### get-training-data
```bash
crowdcent get-training-data <challenge_slug> <version>
```
Get metadata for a specific training dataset version.

#### download-training-data
```bash
crowdcent download-training-data <challenge_slug> <version> [OPTIONS]
```
Download a training dataset file.

**Options:**
- `-o, --output <path>` - Output location (default: `<challenge_slug>_training_v<version>.parquet`)

**Example:**
```bash
crowdcent download-training-data main-challenge 1.0 -o ./data/training.parquet
```

### Inference Data

#### list-inference-data
```bash
crowdcent list-inference-data <challenge_slug>
```
List all inference data periods for the challenge.

#### get-current-inference-data
```bash
crowdcent get-current-inference-data <challenge_slug>
```
Get metadata for the currently active inference period.

#### get-inference-data
```bash
crowdcent get-inference-data <challenge_slug> <release_date>
```
Get metadata for a specific inference period by date (YYYY-MM-DD).

#### download-inference-data
```bash
crowdcent download-inference-data <challenge_slug> <release_date> [OPTIONS]
```
Download inference features for prediction.

**Options:**
- `-o, --output <path>` - Output location (default: `<challenge_slug>_inference_<release_date>.parquet`)

**Example:**
```bash
crowdcent download-inference-data main-challenge current -o ./data/inference.parquet
```

### Submissions

#### list-submissions
```bash
crowdcent list-submissions <challenge_slug> [OPTIONS]
```
List your submissions for a challenge.

**Options:**
- `--period <period>` - Filter by period (`current` or date in YYYY-MM-DD format)

#### get-submission
```bash
crowdcent get-submission <challenge_slug> <submission_id>
```
Get details for a specific submission by ID.

#### submit
```bash
crowdcent submit <challenge_slug> <path_to_predictions>
```
Submit a prediction file to the current inference period.

**Requirements:**
- File must be Parquet format
- Required columns: `id`, `pred_1M`, `pred_3M`, `pred_6M`, `pred_9M`, `pred_12M`

**Example:**
```bash
crowdcent submit main-challenge ./predictions.parquet
```

## Meta Model

#### download-meta-model
```bash
crowdcent download-meta-model <challenge_slug> [OPTIONS]
```
Download the consolidated meta model for a challenge.

**Options:**
- `-o, --output <path>` - Output location (default: `<challenge_slug>_meta_model.parquet`)

**Example:**
```bash
crowdcent download-meta-model main-challenge -o ./data/meta_model.parquet
```

## Error Handling

All errors are printed to stderr. Commands will abort on failure with a non-zero exit code. 