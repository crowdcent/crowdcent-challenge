import click
import logging
import json

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

# --- Helper Functions ---


def get_client(challenge_slug=None):
    """
    Instantiates and returns the ChallengeClient, handling API key loading.

    Args:
        challenge_slug: The challenge slug for client initialization.
                        If None, no client is created (for list_challenges).
    """
    try:
        # Client handles loading API key from env/dotenv
        if challenge_slug:
            return ChallengeClient(challenge_slug=challenge_slug)
        return None
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
@click.argument("challenge_slug", type=str)
@handle_api_error
def get_challenge(challenge_slug):
    """Get details for a specific challenge by slug."""
    client = get_client(challenge_slug)
    challenge = client.get_challenge()
    click.echo(json.dumps(challenge, indent=2))


# --- Training Data Commands ---


@cli.command("list-training-data")
@click.argument("challenge_slug", type=str)
@handle_api_error
def list_training_data(challenge_slug):
    """List all training datasets for a specific challenge."""
    client = get_client(challenge_slug)
    datasets = client.list_training_datasets()
    click.echo(json.dumps(datasets, indent=2))


@cli.command("get-latest-training-data")
@click.argument("challenge_slug", type=str)
@handle_api_error
def get_latest_training_data(challenge_slug):
    """Get the latest training dataset for a specific challenge."""
    client = get_client(challenge_slug)
    dataset = client.get_latest_training_dataset()
    click.echo(json.dumps(dataset, indent=2))


@cli.command("get-training-data")
@click.argument("challenge_slug", type=str)
@click.argument("version", type=str)
@handle_api_error
def get_training_data(challenge_slug, version):
    """Get details for a specific training dataset version."""
    client = get_client(challenge_slug)
    dataset = client.get_training_dataset(version)
    click.echo(json.dumps(dataset, indent=2))


@cli.command("download-training-data")
@click.argument("challenge_slug", type=str)
@click.argument("version", type=str)
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
        dest_path = f"{challenge_slug}_training_v{version}.parquet"

    client.download_training_dataset(version, dest_path)
    click.echo(f"Training data downloaded successfully to {dest_path}")


# --- Inference Data Commands ---


@cli.command("list-inference-data")
@click.argument("challenge_slug", type=str)
@handle_api_error
def list_inference_data(challenge_slug):
    """List all inference data periods for a specific challenge."""
    client = get_client(challenge_slug)
    inference_data = client.list_inference_data()
    click.echo(json.dumps(inference_data, indent=2))


@cli.command("get-current-inference-data")
@click.argument("challenge_slug", type=str)
@handle_api_error
def get_current_inference_data(challenge_slug):
    """Get the currently active inference data period for a specific challenge."""
    client = get_client(challenge_slug)
    inference_data = client.get_current_inference_data()
    click.echo(json.dumps(inference_data, indent=2))


@cli.command("get-inference-data")
@click.argument("challenge_slug", type=str)
@click.argument("release_date", type=str)
@handle_api_error
def get_inference_data(challenge_slug, release_date):
    """Get details for a specific inference data period by release date.

    RELEASE_DATE should be in 'YYYY-MM-DD' format.
    """
    client = get_client(challenge_slug)
    inference_data = client.get_inference_data(release_date)
    click.echo(json.dumps(inference_data, indent=2))


@cli.command("download-inference-data")
@click.argument("challenge_slug", type=str)
@click.argument("release_date", type=str)
@click.option(
    "-o",
    "--output",
    "dest_path",
    default=None,
    help="Output file path. Defaults to [challenge_slug]_inference_[release_date].parquet in current directory.",
)
@handle_api_error
def download_inference_data(challenge_slug, release_date, dest_path):
    """Download the inference features file for a specific period.

    RELEASE_DATE should be in 'YYYY-MM-DD' format or 'current' for the current active period.
    """
    client = get_client(challenge_slug)
    if dest_path is None:
        # Format date part of the filename
        date_str = release_date if release_date != "current" else "current"
        dest_path = f"{challenge_slug}_inference_{date_str}.parquet"

    try:
        client.download_inference_data(release_date, dest_path)
        click.echo(f"Inference data downloaded successfully to {dest_path}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except CrowdCentAPIError as e:
        click.echo(f"Error downloading or writing file: {e}", err=True)
        raise click.Abort()


# --- Submission Commands ---


@cli.command("list-submissions")
@click.argument("challenge_slug", type=str)
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
@click.argument("challenge_slug", type=str)
@click.argument("submission_id", type=int)
@handle_api_error
def get_submission(challenge_slug, submission_id):
    """Get details for a specific submission by ID within a challenge."""
    client = get_client(challenge_slug)
    submission = client.get_submission(submission_id)
    click.echo(json.dumps(submission, indent=2))


@cli.command("submit")
@click.argument("challenge_slug", type=str)
@click.argument(
    "file_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@handle_api_error
def submit(challenge_slug, file_path):
    """Submit a prediction file (Parquet) to a specific challenge.

    The file must be a Parquet file with the required columns:
    id, pred_1M, pred_3M, pred_6M, pred_9M, pred_12M

    The submission will be made to the currently active inference period.
    """
    client = get_client(challenge_slug)
    try:
        submission = client.submit_predictions(file_path)
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
@click.argument("challenge_slug", type=str)
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
        dest_path = f"{challenge_slug}_meta_model.parquet"

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
