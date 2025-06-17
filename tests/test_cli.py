import pytest
import json
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import polars as pl

# Assuming cli.py is in src/crowdcent_challenge/
from crowdcent_challenge.cli import cli
from crowdcent_challenge.client import (
    ChallengeClient,
    AuthenticationError,
    NotFoundError,
    ClientError,
)

TEST_SLUG = "test-challenge"
TEST_API_KEY = "cli_test_key"

# --- Fixtures ---


@pytest.fixture
def runner():
    """Provides a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def mock_client():
    """Provides a MagicMock substitute for ChallengeClient."""
    mock = MagicMock(spec=ChallengeClient)
    # Set the challenge_slug attribute
    mock.challenge_slug = TEST_SLUG
    # Set default return values for methods that return dicts/lists
    mock.get_challenge.return_value = {"name": "Mock Challenge", "slug": TEST_SLUG}
    mock.list_training_datasets.return_value = []
    mock.get_training_dataset.return_value = {}
    mock.list_inference_data.return_value = []
    mock.get_inference_data.return_value = {}
    mock.list_submissions.return_value = []
    mock.get_submission.return_value = {}
    mock.submit_predictions.return_value = {}
    # Make download methods do nothing by default (can be overridden per test)
    mock.download_training_dataset.return_value = None
    mock.download_inference_data.return_value = None
    return mock


@pytest.fixture(autouse=True)
def patch_get_client(mock_client):
    """Automatically replaces the client instance used by CLI commands."""
    # Patch the helper function that creates the client instance
    with patch(
        "crowdcent_challenge.cli.get_client", return_value=mock_client
    ) as patched:
        # Special handling for list-challenges which doesn't need a slug/client instance initially
        # We patch the *class method* called by the command
        with patch.object(ChallengeClient, "list_all_challenges") as mock_list_all:
            mock_list_all.return_value = []  # Default empty list
            yield patched, mock_list_all


@pytest.fixture
def mock_predictions_file(tmp_path):
    """Creates a dummy Parquet prediction file."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "pred_1M": [0.1, 0.2, 0.3],
            "pred_3M": [0.1, 0.2, 0.3],
            "pred_6M": [0.1, 0.2, 0.3],
            "pred_9M": [0.1, 0.2, 0.3],
            "pred_12M": [0.1, 0.2, 0.3],
        }
    )
    file_path = tmp_path / "preds.parquet"
    df.write_parquet(file_path)
    return str(file_path)


# --- Basic CLI Tests ---


def test_cli_help(runner):
    """Test the main help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Command Line Interface for the CrowdCent Challenge." in result.output
    assert "list-challenges" in result.output
    assert "submit" in result.output


# --- Challenge Command Tests ---


def test_list_challenges_success(runner, patch_get_client):
    """Test successful listing of challenges."""
    _, mock_list_all = patch_get_client
    mock_data = [
        {"name": "Challenge A", "slug": "a"},
        {"name": "Challenge B", "slug": "b"},
    ]
    mock_list_all.return_value = mock_data

    result = runner.invoke(cli, ["list-challenges"])
    print(result.output)
    assert result.exit_code == 0
    assert json.loads(result.output) == mock_data
    mock_list_all.assert_called_once()


def test_list_challenges_auth_error(runner, patch_get_client):
    """Test auth error when listing challenges."""
    _, mock_list_all = patch_get_client
    mock_list_all.side_effect = AuthenticationError("Bad key")

    result = runner.invoke(cli, ["list-challenges"])
    assert result.exit_code != 0  # Abort raises SystemExit
    assert "Error: Authentication failed" in result.output
    assert "Bad key" in result.output


def test_get_challenge_success(runner, mock_client):
    """Test successful getting of a specific challenge."""
    mock_data = {
        "name": "Mock Challenge",
        "slug": TEST_SLUG,
        "description": "Details...",
    }
    mock_client.get_challenge.return_value = mock_data

    result = runner.invoke(cli, ["get-challenge", "--challenge", TEST_SLUG])
    assert result.exit_code == 0
    assert json.loads(result.output) == mock_data
    mock_client.get_challenge.assert_called_once()


def test_get_challenge_not_found(runner, mock_client):
    """Test challenge not found error."""
    mock_client.get_challenge.side_effect = NotFoundError("No such challenge")
    result = runner.invoke(cli, ["get-challenge", "--challenge", "nonexistent-slug"])
    assert result.exit_code != 0
    assert "Error: Resource not found." in result.output
    assert "No such challenge" in result.output


# --- Training Data Command Tests ---


def test_list_training_data_success(runner, mock_client):
    mock_data = [
        {"version": "1.0", "is_latest": True},
        {"version": "0.9", "is_latest": False},
    ]
    mock_client.list_training_datasets.return_value = mock_data
    result = runner.invoke(cli, ["list-training-data", "--challenge", TEST_SLUG])
    assert result.exit_code == 0
    assert json.loads(result.output) == mock_data
    mock_client.list_training_datasets.assert_called_once()


def test_download_training_data_success(runner, mock_client, tmp_path):
    output_file = tmp_path / "training_data.parquet"
    version = "1.0"
    result = runner.invoke(
        cli,
        [
            "download-training-data",
            "--challenge",
            TEST_SLUG,
            version,
            "-o",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert f"Training data downloaded successfully to {output_file}" in result.output
    mock_client.download_training_dataset.assert_called_once_with(
        version, str(output_file)
    )


def test_download_training_data_latest(runner, mock_client, tmp_path):
    output_file = tmp_path / "training_latest.parquet"
    version = "latest"
    result = runner.invoke(
        cli,
        [
            "download-training-data",
            "--challenge",
            TEST_SLUG,
            version,
            "-o",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert f"Training data downloaded successfully to {output_file}" in result.output
    mock_client.download_training_dataset.assert_called_once_with(
        version, str(output_file)
    )


def test_download_training_data_default_output(
    runner, mock_client, tmp_path, monkeypatch
):
    # Run command in tmp_path so default file is created there
    monkeypatch.chdir(tmp_path)
    version = "1.1"
    expected_relative_output = f"{TEST_SLUG}_training_v{version}.parquet"
    result = runner.invoke(
        cli, ["download-training-data", "--challenge", TEST_SLUG, version]
    )
    assert result.exit_code == 0
    # Assert based on the relative path generated by the CLI
    mock_client.download_training_dataset.assert_called_once_with(
        version, expected_relative_output
    )


def test_download_training_data_api_error(runner, mock_client, tmp_path):
    output_file = tmp_path / "training_data.parquet"
    version = "1.0"
    mock_client.download_training_dataset.side_effect = NotFoundError(
        "Dataset version not found"
    )
    result = runner.invoke(
        cli,
        [
            "download-training-data",
            "--challenge",
            TEST_SLUG,
            version,
            "-o",
            str(output_file),
        ],
    )
    assert result.exit_code != 0
    # Error message comes from the main decorator now
    assert "Error: Resource not found." in result.output
    assert "Dataset version not found" in result.output


# --- Submission Command Tests ---


def test_submit_success(runner, mock_client, mock_predictions_file):
    mock_response = {"id": 123, "status": "pending", "submitted_at": "..."}
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli, ["submit", "--challenge", TEST_SLUG, mock_predictions_file]
    )
    assert result.exit_code == 0
    assert "Submission successful!" in result.output
    assert (
        json.loads(result.output.split("\n", 1)[1]) == mock_response
    )  # Check JSON part
    mock_client.submit_predictions.assert_called_once_with(
        mock_predictions_file, slot=1
    )


def test_submit_file_not_found(runner):
    # Click's path type validation happens before our command, so expect Click's error
    result = runner.invoke(
        cli, ["submit", "--challenge", TEST_SLUG, "nonexistent.parquet"]
    )
    assert result.exit_code != 0
    # This is Click's standard error format for missing files
    assert "Error: Invalid value for 'FILE_PATH'" in result.output
    assert "nonexistent.parquet" in result.output


def test_submit_api_error(runner, mock_client, mock_predictions_file):
    # The CLI's submit command has a specific error handler: click.echo(f"Error during submission: {e}", err=True)
    # When a ClientError is created, its string representation is just the message passed to it
    mock_client.submit_predictions.side_effect = ClientError(
        "Invalid submission format"
    )
    result = runner.invoke(
        cli, ["submit", "--challenge", TEST_SLUG, mock_predictions_file]
    )
    assert result.exit_code != 0
    # Match the exact format: "Error during submission: " + str(ClientError)
    assert "Error during submission: Invalid submission format" in result.output


# --- TODO: Add more tests for other CLI commands and edge cases ---
