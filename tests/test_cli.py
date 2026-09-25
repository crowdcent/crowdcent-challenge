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
    # JSON is after the message lines
    json_output = result.output.split("\n", 1)[1]
    assert json.loads(json_output) == mock_response
    mock_client.submit_predictions.assert_called_once_with(
        mock_predictions_file,
        slot=1,
        queue_next=True,
        is_experimental=False,
        notes="",
    )


def test_submit_success_with_auto_queue(runner, mock_client, mock_predictions_file):
    """Test that auto-queue message is shown when queued_for_next is true."""
    mock_response = {"id": 123, "status": "pending", "queued_for_next": True}
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli, ["submit", "--challenge", TEST_SLUG, mock_predictions_file]
    )
    assert result.exit_code == 0
    assert "Submission successful!" in result.output
    assert "Also queued for next period." in result.output
    # Verify JSON parsing still works with merged status line
    json_output = result.output.split("\n", 1)[1]
    assert json.loads(json_output) == mock_response


def test_submit_no_queue_next(runner, mock_client, mock_predictions_file):
    """Test --no-queue-next flag is passed to client."""
    mock_response = {"id": 123, "status": "pending"}
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli,
        ["submit", "--challenge", TEST_SLUG, "--no-queue-next", mock_predictions_file],
    )
    assert result.exit_code == 0
    assert "Submission successful!" in result.output
    mock_client.submit_predictions.assert_called_once_with(
        mock_predictions_file,
        slot=1,
        queue_next=False,
        is_experimental=False,
        notes="",
    )


def test_submit_queued_response(runner, mock_client, mock_predictions_file):
    """Test CLI handles queued response (when no active window)."""
    mock_response = {
        "status": "queued",
        "slot": 1,
        "message": "Submission queued for slot 1.",
        "is_experimental": False,
        "notes": "",
    }
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli, ["submit", "--challenge", TEST_SLUG, mock_predictions_file]
    )
    assert result.exit_code == 0
    assert "Submission queued for next period." in result.output
    assert "Submission successful!" not in result.output


def test_submit_experimental_with_notes(runner, mock_client, mock_predictions_file):
    """--experimental and --notes are forwarded to the client and reflected in output."""
    mock_response = {
        "id": 555,
        "status": "pending",
        "is_experimental": True,
        "notes": "transformer w/ sector embeddings",
        "queued_for_next": True,
    }
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli,
        [
            "submit",
            "--challenge",
            TEST_SLUG,
            "--slot",
            "2",
            "--experimental",
            "--notes",
            "transformer w/ sector embeddings",
            mock_predictions_file,
        ],
    )
    assert result.exit_code == 0
    assert "Submission successful!" in result.output
    assert "Marked as experimental." in result.output
    mock_client.submit_predictions.assert_called_once_with(
        mock_predictions_file,
        slot=2,
        queue_next=True,
        is_experimental=True,
        notes="transformer w/ sector embeddings",
    )


def test_submit_partial_success_queue_rejected(
    runner, mock_client, mock_predictions_file
):
    """Live save succeeded but queue copy rejected: surface the queue_error_code."""
    mock_response = {
        "id": 777,
        "status": "pending",
        "is_experimental": True,
        "queued_for_next": False,
        "queue_error_code": "EXPERIMENTAL_QUEUE_REQUIRES_NON_EXP",
        "queue_error": "Queue a non-experimental prediction first.",
    }
    mock_client.submit_predictions.return_value = mock_response

    result = runner.invoke(
        cli,
        [
            "submit",
            "--challenge",
            TEST_SLUG,
            "--experimental",
            mock_predictions_file,
        ],
    )
    assert result.exit_code == 0
    assert "Submission successful!" in result.output
    assert (
        "Queue copy was rejected (EXPERIMENTAL_QUEUE_REQUIRES_NON_EXP)" in result.output
    )
    assert "Also queued for next period." not in result.output


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


# --- Help text ---


def test_commands_keep_their_help_text(runner):
    """handle_api_error must not swallow the docstring click shows as help."""
    result = runner.invoke(cli, ["--help"])
    assert "Submit a prediction file" in result.output
    result = runner.invoke(cli, ["submit", "--help"])
    assert "Parquet or CSV" in result.output


# --- Account ---


def test_whoami(runner, mock_client):
    mock_client.check_auth.return_value = {"username": "me", "allow_cloud": True}
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 0
    assert json.loads(result.output)["username"] == "me"


def test_performance_filters(runner, mock_client):
    mock_client.get_performance.return_value = [{"id": 1}]
    result = runner.invoke(cli, ["performance", "--slot", "2", "--include-pending"])
    assert result.exit_code == 0
    mock_client.get_performance.assert_called_once_with(scored_only=False, slot=2)


# --- Simulator ---


def test_sim_run_inline_config(runner, mock_client):
    mock_client.run_simulation.return_value = {"oos_stats": {"sharpe": 1.2}}
    result = runner.invoke(
        cli,
        [
            "sim",
            "run",
            "--config",
            '{"n_long": 10}',
            "--include",
            "curve",
            "--leverage",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.run_simulation.assert_called_once_with(
        {"n_long": 10},
        config_token=None,
        include=["curve"],
        benchmark_trials=0,
        leverage=2.0,
        target_vol=0.0,
    )


def test_sim_run_config_from_file(runner, mock_client, tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"n_short": 5}')
    mock_client.run_simulation.return_value = {}
    result = runner.invoke(cli, ["sim", "run", "--config", f"@{config}"])
    assert result.exit_code == 0, result.output
    assert mock_client.run_simulation.call_args.args[0] == {"n_short": 5}


def test_sim_run_needs_exactly_one_config(runner, mock_client):
    result = runner.invoke(cli, ["sim", "run"])
    assert result.exit_code != 0
    result = runner.invoke(cli, ["sim", "run", "--config", "{}", "--config-token", "t"])
    assert result.exit_code != 0
    mock_client.run_simulation.assert_not_called()


def test_sim_run_rejects_bad_json(runner, mock_client):
    result = runner.invoke(cli, ["sim", "run", "--config", "{n_long: 10}"])
    assert result.exit_code == 2
    assert "not valid JSON" in result.output
    mock_client.run_simulation.assert_not_called()


def test_sim_sweep_and_blend(runner, mock_client):
    mock_client.run_sweep.return_value = {"results": []}
    mock_client.run_blend.return_value = {"stats": {}}
    result = runner.invoke(
        cli,
        [
            "sim",
            "sweep",
            "--config",
            '{"n_short": 10}',
            "--sweep",
            '{"n_long": [5, 10]}',
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.run_sweep.assert_called_once_with({"n_short": 10}, {"n_long": [5, 10]})

    sleeves = '[{"config": {"n_long": 5}, "weight": 1.0}]'
    result = runner.invoke(
        cli, ["sim", "blend", "--sleeves", sleeves, "--target-vol", "0.2"]
    )
    assert result.exit_code == 0, result.output
    mock_client.run_blend.assert_called_once_with(
        [{"config": {"n_long": 5}, "weight": 1.0}], leverage=1.0, target_vol=0.2
    )


# --- Trading ---


def test_trade_defaults_to_testnet(runner, mock_client):
    mock_client.get_mandate.return_value = {}
    runner.invoke(cli, ["trade", "mandate"])
    mock_client.get_mandate.assert_called_once_with(network="testnet")


def test_trade_preview_prints_the_execute_command(runner, mock_client):
    mock_client.preview_rebalance.return_value = {"plan_hash": "abc", "trades": []}
    result = runner.invoke(cli, ["trade", "preview", "--network", "mainnet"])
    assert result.exit_code == 0, result.output
    assert "crowdcent trade execute abc --network mainnet" in result.output


def test_trade_execute_aborts_without_confirmation(runner, mock_client):
    result = runner.invoke(cli, ["trade", "execute", "abc"], input="n\n")
    assert result.exit_code != 0
    mock_client.execute_rebalance.assert_not_called()


def test_trade_execute_aborts_without_a_terminal(runner, mock_client):
    """An agent with no stdin must pass --yes; a closed prompt never executes."""
    result = runner.invoke(cli, ["trade", "execute", "abc"], input="")
    assert result.exit_code != 0
    mock_client.execute_rebalance.assert_not_called()


def test_trade_execute_after_confirmation(runner, mock_client):
    mock_client.execute_rebalance.return_value = {"status": "done"}
    result = runner.invoke(cli, ["trade", "execute", "abc"], input="y\n")
    assert result.exit_code == 0, result.output
    mock_client.execute_rebalance.assert_called_once_with("abc", network="testnet")


def test_trade_execute_with_yes(runner, mock_client):
    mock_client.execute_rebalance.return_value = {}
    result = runner.invoke(
        cli, ["trade", "execute", "abc", "--yes", "--network", "mainnet"]
    )
    assert result.exit_code == 0, result.output
    mock_client.execute_rebalance.assert_called_once_with("abc", network="mainnet")


def test_trade_flatten_previews_without_a_hash(runner, mock_client):
    mock_client.flatten.return_value = {"plan_hash": "f1"}
    result = runner.invoke(cli, ["trade", "flatten"])
    assert result.exit_code == 0, result.output
    mock_client.flatten.assert_called_once_with(preview=True, network="testnet")
    assert "crowdcent trade flatten f1" in result.output


def test_trade_flatten_executes_with_a_hash(runner, mock_client):
    mock_client.flatten.return_value = {}
    result = runner.invoke(cli, ["trade", "flatten", "f1", "-y"])
    assert result.exit_code == 0, result.output
    mock_client.flatten.assert_called_once_with("f1", network="testnet")


def test_trade_pause_needs_no_confirmation(runner, mock_client):
    mock_client.pause_trading.return_value = {"paused": True}
    result = runner.invoke(cli, ["trade", "pause"], input="")
    assert result.exit_code == 0, result.output
    mock_client.pause_trading.assert_called_once_with(network="testnet")


def test_trade_set_mandate_confirms(runner, mock_client):
    mock_client.set_mandate.return_value = {}
    result = runner.invoke(
        cli, ["trade", "set-mandate", "--mandate", '{"sleeves": []}'], input=""
    )
    assert result.exit_code != 0
    mock_client.set_mandate.assert_not_called()
    result = runner.invoke(
        cli, ["trade", "set-mandate", "--mandate", '{"sleeves": []}', "-y"]
    )
    assert result.exit_code == 0, result.output
    mock_client.set_mandate.assert_called_once_with({"sleeves": []}, network="testnet")


# --- Cloud ---


def test_cloud_create_from_recipe(runner, mock_client):
    mock_client.create_cloud_project.return_value = {"id": "p1"}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "create",
            "My model",
            "--recipe",
            "hyperliquid-ranking",
            "--challenge-access",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.create_cloud_project.assert_called_once_with(
        "My model",
        source=None,
        filename="notebook.py",
        files=None,
        recipe="hyperliquid-ranking",
        challenge_access=True,
    )


def test_cloud_create_from_local_files(runner, mock_client, tmp_path):
    main = tmp_path / "train.py"
    main.write_text("print('train')")
    helper = tmp_path / "helpers.py"
    helper.write_text("X = 1")
    mock_client.create_cloud_project.return_value = {"id": "p1"}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "create",
            "M",
            "--source",
            str(main),
            "--file",
            f"lib/helpers.py={helper}",
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.create_cloud_project.call_args.kwargs
    assert kwargs["source"] == "print('train')"
    assert kwargs["filename"] == "train.py"
    assert kwargs["files"] == {"lib/helpers.py": "X = 1"}


def test_cloud_create_needs_recipe_or_source(runner, mock_client):
    result = runner.invoke(cli, ["cloud", "create", "M"])
    assert result.exit_code != 0
    mock_client.create_cloud_project.assert_not_called()


def test_cloud_save_edits_and_deletes(runner, mock_client, tmp_path):
    local = tmp_path / "predict.py"
    local.write_text("new")
    mock_client.update_cloud_project.return_value = {"latest_version": 4}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "save",
            "p1",
            f"predict.py={local}",
            "--base-version",
            "3",
            "--delete",
            "old.py",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.update_cloud_project.assert_called_once_with(
        "p1", base_version=3, files={"predict.py": "new", "old.py": None}
    )


def test_cloud_upload_maps_paths(runner, mock_client, tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"\x00")
    mock_client.upload_cloud_project_files.return_value = {}
    result = runner.invoke(cli, ["cloud", "upload", "p1", f"models/model.pkl={model}"])
    assert result.exit_code == 0, result.output
    mock_client.upload_cloud_project_files.assert_called_once_with(
        "p1", {"models/model.pkl": str(model)}
    )


def test_cloud_upload_missing_file(runner, mock_client):
    result = runner.invoke(cli, ["cloud", "upload", "p1", "nope.pkl"])
    assert result.exit_code == 2
    mock_client.upload_cloud_project_files.assert_not_called()


def test_cloud_run_parses_params(runner, mock_client):
    mock_client.run_cloud_project.return_value = {"id": "r1", "state": "queued"}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "run",
            "p1",
            "--entrypoint",
            "train.py",
            "--envelope",
            "m",
            "--param",
            "trials=100",
            "--param",
            "fast=true",
            "--param",
            "name=lgbm",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.run_cloud_project.assert_called_once_with(
        "p1",
        version=None,
        envelope="m",
        time_limit_minutes=None,
        entrypoint="train.py",
        idempotency_key=None,
        parameters={"trials": 100, "fast": True, "name": "lgbm"},
    )


def test_cloud_run_wait_polls_to_the_end(runner, mock_client):
    mock_client.run_cloud_project.return_value = {"id": "r1", "state": "queued"}
    mock_client.get_cloud_run.side_effect = [
        {"id": "r1", "state": "running", "detail": ""},
        {"id": "r1", "state": "done", "detail": "finished"},
    ]
    with patch("crowdcent_challenge.cli.time.sleep"):
        result = runner.invoke(cli, ["cloud", "run", "p1", "--wait"])
    assert result.exit_code == 0, result.output
    assert mock_client.get_cloud_run.call_count == 2
    assert '"state": "done"' in result.output


def test_cloud_run_wait_fails_on_a_failed_run(runner, mock_client):
    mock_client.run_cloud_project.return_value = {"id": "r1", "state": "queued"}
    mock_client.get_cloud_run.return_value = {
        "id": "r1",
        "state": "failed",
        "detail": "boom",
    }
    result = runner.invoke(cli, ["cloud", "run", "p1", "--wait"])
    assert result.exit_code == 1


def test_cloud_schedule_on_release(runner, mock_client):
    mock_client.schedule_cloud_project.return_value = {"next_due": None}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "schedule",
            "p1",
            "--trigger",
            "on_inference_release",
            "--release-challenge",
            "hyperliquid-ranking",
            "--entrypoint",
            "predict.py",
            "--follow-head",
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.schedule_cloud_project.call_args.kwargs
    assert kwargs["trigger"] == "on_inference_release"
    assert kwargs["challenge"] == "hyperliquid-ranking"
    assert kwargs["entrypoint"] == "predict.py"
    assert kwargs["follow_head"] is True
    assert kwargs["run_id"] is None


def test_cloud_unschedule_and_stop(runner, mock_client):
    mock_client.pause_cloud_project_schedule.return_value = {"paused": True}
    mock_client.stop_cloud_run.return_value = {"state": "canceled"}
    assert runner.invoke(cli, ["cloud", "unschedule", "p1"]).exit_code == 0
    mock_client.pause_cloud_project_schedule.assert_called_once_with(
        "p1", entrypoint=None
    )
    assert runner.invoke(cli, ["cloud", "stop", "r1"]).exit_code == 0
    mock_client.stop_cloud_run.assert_called_once_with("r1")


def test_cloud_archive_confirms(runner, mock_client):
    result = runner.invoke(cli, ["cloud", "archive", "p1"], input="")
    assert result.exit_code != 0
    mock_client.archive_cloud_project.assert_not_called()


def test_cloud_api_errors_are_reported(runner, mock_client):
    mock_client.get_cloud_project.side_effect = NotFoundError("no such project")
    result = runner.invoke(cli, ["cloud", "project", "p1"])
    assert result.exit_code != 0
    assert "no such project" in result.output


def test_cloud_update_settings(runner, mock_client):
    mock_client.update_cloud_project.return_value = {}
    result = runner.invoke(
        cli,
        [
            "cloud",
            "update",
            "p1",
            "--name",
            "New",
            "--no-challenge-access",
            "--history-keep",
            "all",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.update_cloud_project.assert_called_once_with(
        "p1", name="New", challenge_access=False, history_keep=None
    )


def test_cloud_update_prune_confirms_and_stands_alone(runner, mock_client):
    mock_client.update_cloud_project.return_value = {}
    result = runner.invoke(cli, ["cloud", "update", "p1", "--prune-history"], input="")
    assert result.exit_code != 0
    result = runner.invoke(
        cli, ["cloud", "update", "p1", "--prune-history", "--name", "x", "-y"]
    )
    assert result.exit_code == 2
    mock_client.update_cloud_project.assert_not_called()
    result = runner.invoke(cli, ["cloud", "update", "p1", "--prune-history", "-y"])
    assert result.exit_code == 0, result.output
    mock_client.update_cloud_project.assert_called_once_with("p1", prune_history=True)
