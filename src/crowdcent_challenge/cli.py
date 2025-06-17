import click
import logging
import json
import os
from pathlib import Path

from .client import (
    ChallengeClient,
    CrowdCentAPIError,
    AuthenticationError,
    NotFoundError,
    ClientError,
    ServerError,
)

# Configure basic logging for the CLI
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Config Functions ---


def get_config_dir():
    """Return the directory for storing crowdcent-challenge configuration."""
    if os.name == "nt":  # Windows
        config_dir = Path(os.environ.get("APPDATA", "")) / "crowdcent-challenge"
    else:  # Unix/Linux/Mac
        config_dir = Path.home() / ".config" / "crowdcent-challenge"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file():
    """Return the path to the configuration file."""
    return get_config_dir() / "config.json"


def load_config():
    """Load configuration from file."""
    config_file = get_config_file()
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Error parsing config file {config_file}. Using default configuration."
            )

    return {}


def save_config(config):
    """Save configuration to file."""
    config_file = get_config_file()
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def get_default_challenge_slug():
    """Get the default challenge slug from the configuration."""
    config = load_config()
    return config.get("default_challenge_slug")


# --- Helper Functions ---


def get_client(challenge_slug=None):
    """
    Instantiates and returns the ChallengeClient, handling API key loading.

    Args:
        challenge_slug: The challenge slug for client initialization.
                        If None, tries to use the default challenge.
    """
    try:
        # If no challenge slug provided, try to use default
        if challenge_slug is None:
            challenge_slug = get_default_challenge_slug()
            if challenge_slug is None:
                # Try to auto-select if only one challenge exists
                try:
                    challenges = ChallengeClient.list_all_challenges()
                    if len(challenges) == 1:
                        challenge_slug = challenges[0]["slug"]
                        click.echo(
                            f"Auto-selected challenge: {challenge_slug}", err=True
                        )
                    elif len(challenges) > 1:
                        click.echo(
                            "Multiple challenges available. Please specify one:",
                            err=True,
                        )
                        for c in challenges:
                            click.echo(f"  - {c['slug']}: {c['name']}", err=True)
                        raise click.UsageError(
                            "Use --challenge option or set a default with 'crowdcent set-default-challenge'."
                        )
                    else:
                        raise click.UsageError("No challenges available.")
                except AuthenticationError:
                    raise click.UsageError(
                        "Cannot list challenges. Check your API key configuration."
                    )

        # Client handles loading API key from env/dotenv
        return ChallengeClient(challenge_slug=challenge_slug)
    except AuthenticationError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def handle_api_error(func):
    """Decorator to catch and handle common API errors for CLI commands."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotFoundError as e:
            click.echo(f"Error: Resource not found. {e}", err=True)
        except AuthenticationError as e:
            click.echo(f"Error: Authentication failed. Check API key. {e}", err=True)
        except ClientError as e:
            click.echo(f"Error: Client error (e.g., bad request). {e}", err=True)
        except ServerError as e:
            click.echo(f"Error: Server error. Please try again later. {e}", err=True)
        except CrowdCentAPIError as e:
            click.echo(f"Error: API call failed. {e}", err=True)
        except Exception as e:
            click.echo(f"An unexpected error occurred: {e}", err=True)
            logger.exception(
                "Unexpected CLI error"
            )  # Log traceback for unexpected errors
        raise click.Abort()

    return wrapper


# --- CLI Commands ---


@click.group()
def cli():
    """Command Line Interface for the CrowdCent Challenge."""
    pass


# --- Challenge Commands ---


@cli.command("set-default-challenge")
@click.argument("challenge_slug", type=str)
@handle_api_error
def set_default_challenge(challenge_slug):
    """Set the default challenge slug for future commands.

    This is optional - if you only have one challenge, it will be auto-selected.
    Setting a default is useful when working with multiple challenges.
    """
    # Verify the challenge exists
    try:
        client = get_client(challenge_slug)
        client.get_challenge()  # Check if challenge exists

        config = load_config()
        config["default_challenge_slug"] = challenge_slug
        save_config(config)

        click.echo(f"Default challenge set to '{challenge_slug}'")
    except Exception as e:
        click.echo(f"Error setting default challenge: {e}", err=True)
        raise click.Abort()


@cli.command("get-default-challenge")
def get_default_challenge():
    """Show the current default challenge slug."""
    challenge_slug = get_default_challenge_slug()
    if challenge_slug:
        click.echo(f"Current default challenge: {challenge_slug}")
    else:
        click.echo(
            "No default challenge set. Use 'crowdcent set-default-challenge' to set one."
        )


@cli.command("list-challenges")
@handle_api_error
def list_challenges():
    """List all active challenges."""
    try:
        # Use the class method directly - no client instance needed
        challenges = ChallengeClient.list_all_challenges()
        click.echo(json.dumps(challenges, indent=2))
    except AuthenticationError as e:
        click.echo(f"Error: Authentication failed. Check API key. {e}", err=True)
        raise click.Abort()


@cli.command("get-challenge")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@handle_api_error
def get_challenge(challenge_slug):
    """Get details for a specific challenge by slug."""
    client = get_client(challenge_slug)
    challenge = client.get_challenge()
    click.echo(json.dumps(challenge, indent=2))


# --- Training Data Commands ---


@cli.command("list-training-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@handle_api_error
def list_training_data(challenge_slug):
    """List all training datasets for a specific challenge."""
    client = get_client(challenge_slug)
    datasets = client.list_training_datasets()
    click.echo(json.dumps(datasets, indent=2))


@cli.command("get-training-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument("version", type=str, default="latest")
@handle_api_error
def get_training_data(challenge_slug, version):
    """Get details for a specific training dataset version."""
    client = get_client(challenge_slug)
    dataset = client.get_training_dataset(version)
    click.echo(json.dumps(dataset, indent=2))


@cli.command("download-training-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument("version", type=str, default="latest")
@click.option(
    "-o",
    "--output",
    "dest_path",
    default=None,
    help="Output file path. Defaults to [challenge_slug]_training_v[version].parquet in current directory.",
)
@handle_api_error
def download_training_data(challenge_slug, version, dest_path):
    """Download the training data file for a specific challenge and version.

    VERSION can be a specific version string (e.g., '1.0') or 'latest' for the latest version.
    """
    client = get_client(challenge_slug)
    if dest_path is None:
        dest_path = f"{client.challenge_slug}_training_v{version}.parquet"

    client.download_training_dataset(version, dest_path)
    click.echo(f"Training data downloaded successfully to {dest_path}")


# --- Inference Data Commands ---


@cli.command("list-inference-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@handle_api_error
def list_inference_data(challenge_slug):
    """List all inference data periods for a specific challenge."""
    client = get_client(challenge_slug)
    inference_data = client.list_inference_data()
    click.echo(json.dumps(inference_data, indent=2))


@cli.command("get-inference-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument("release_date", type=str)
@handle_api_error
def get_inference_data(challenge_slug, release_date):
    """Get details for a specific inference data period by release date.

    RELEASE_DATE should be in 'YYYY-MM-DD' format or can be 'current' for the current active period or 'latest' for the latest available period.
    """
    client = get_client(challenge_slug)
    inference_data = client.get_inference_data(release_date)
    click.echo(json.dumps(inference_data, indent=2))


@cli.command("download-inference-data")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument("release_date", default="latest", type=str)
@click.option(
    "-o",
    "--output",
    "dest_path",
    default=None,
    help="Output file path. Defaults to [challenge_slug]_inference_[release_date].parquet in current directory.",
)
# Polling controls
@click.option(
    "--no-poll",
    is_flag=True,
    help="Disable polling when waiting for the current inference data to be published.",
)
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    show_default=True,
    help="Seconds to wait between polling attempts when release_date='current'.",
)
@click.option(
    "--timeout",
    type=int,
    default=900,
    show_default=True,
    help="Maximum seconds to wait when polling for current data (0 = wait indefinitely).",
)
@handle_api_error
def download_inference_data(
    challenge_slug,
    release_date,
    dest_path,
    no_poll,
    poll_interval,
    timeout,
):
    """Download the inference features file for a specific period.

    RELEASE_DATE may be:
    • An explicit date in 'YYYY-MM-DD' format
    • 'current' – the ongoing inference period (will poll by default)
    • 'latest' – most recent published period
    """

    client = get_client(challenge_slug)

    # Default output path
    if dest_path is None:
        date_str = release_date if release_date != "current" else "current"
        dest_path = f"{client.challenge_slug}_inference_{date_str}.parquet"

    # Translate CLI options → client arguments
    poll = not no_poll
    timeout_val = None if timeout == 0 else timeout

    try:
        client.download_inference_data(
            release_date,
            dest_path,
            poll=poll,
            poll_interval=poll_interval,
            timeout=timeout_val,
        )
        click.echo(f"Inference data downloaded successfully to {dest_path}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except CrowdCentAPIError as e:
        click.echo(f"Error downloading or writing file: {e}", err=True)
        raise click.Abort()


# --- Submission Commands ---


@cli.command("list-submissions")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.option(
    "--period",
    type=str,
    help="Filter submissions by period: 'current' or a date in 'YYYY-MM-DD' format",
)
@handle_api_error
def list_submissions(challenge_slug, period):
    """List submissions for a specific challenge with optional period filtering."""
    client = get_client(challenge_slug)
    submissions = client.list_submissions(period)
    click.echo(json.dumps(submissions, indent=2))


@cli.command("get-submission")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument("submission_id", type=int)
@handle_api_error
def get_submission(challenge_slug, submission_id):
    """Get details for a specific submission by ID within a challenge."""
    client = get_client(challenge_slug)
    submission = client.get_submission(submission_id)
    click.echo(json.dumps(submission, indent=2))


@cli.command("submit")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.argument(
    "file_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--slot",
    type=int,
    default=1,
    help="Submission slot number (1-based). Defaults to 1.",
)
@handle_api_error
def submit(challenge_slug, file_path, slot):
    """Submit a prediction file (Parquet) to a specific challenge.

    The file must be a Parquet file with the required columns specified by the challenge
    (e.g., id, pred_10d, pred_30d).

    The submission will be made to the currently active inference period.
    Use --slot to specify a submission slot (1-based).
    """
    client = get_client(challenge_slug)
    try:
        submission = client.submit_predictions(file_path, slot=slot)
        click.echo("Submission successful!")
        click.echo(json.dumps(submission, indent=2))
    except FileNotFoundError:  # Should be caught by click.Path, but handle just in case
        click.echo(f"Error: Prediction file not found at {file_path}", err=True)
        raise click.Abort()
    except CrowdCentAPIError as e:
        click.echo(f"Error during submission: {e}", err=True)
        raise click.Abort()


# --- Meta Model Commands ---


@cli.command("download-meta-model")
@click.option(
    "--challenge",
    "-c",
    "challenge_slug",
    type=str,
    help="Challenge slug (uses default if not specified)",
)
@click.option(
    "-o",
    "--output",
    "dest_path",
    default=None,
    help="Output file path. Defaults to [challenge_slug]_meta_model.parquet in current directory.",
)
@handle_api_error
def download_meta_model(challenge_slug, dest_path):
    """Download the consolidated meta model for a specific challenge.

    The meta model is typically an aggregation (e.g., average) of all valid
    submissions for past inference periods.
    """
    client = get_client(challenge_slug)
    if dest_path is None:
        dest_path = f"{client.challenge_slug}_meta_model.parquet"

    try:
        client.download_meta_model(dest_path)
        click.echo(f"Consolidated meta model downloaded successfully to {dest_path}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except CrowdCentAPIError as e:
        click.echo(f"Error downloading or writing file: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
