"""Challenge/data/submission tools.

Challenge-scoped tools take an explicit ``challenge_slug`` (default
hyperliquid-ranking). Local-filesystem tools register only under stdio;
signed-URL twins work everywhere. Client exceptions propagate; fastmcp
surfaces their messages verbatim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .params import READ_ONLY, ChallengeSlug
from .runtime import (
    DEFAULT_CHALLENGE,
    api_base_url,
    api_key_for_request,
    client_for,
    is_hosted,
)


def _coerced_path(dest_path: str, suffix: str = ".parquet") -> Path:
    path = Path(dest_path).expanduser().resolve()
    if path.suffix not in (".parquet", ".csv"):
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def register_challenge_tools(mcp) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def list_all_challenges() -> List[Dict[str, Any]]:
        """List all active CrowdCent challenges.

        Use this to discover challenge slugs before calling
        challenge-scoped tools.
        """
        from crowdcent_challenge import ChallengeClient

        return ChallengeClient.list_all_challenges(
            api_key=api_key_for_request(), base_url=api_base_url()
        )

    @mcp.tool(annotations=READ_ONLY)
    def get_challenge_info(
        challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Get detailed information about a challenge.

        Args:
            challenge_slug: The challenge to query (default hyperliquid-ranking).
        """
        return client_for(challenge_slug).get_challenge()

    @mcp.tool(annotations=READ_ONLY)
    def list_training_datasets(
        challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE,
    ) -> List[Dict[str, Any]]:
        """List all training dataset versions for a challenge."""
        return client_for(challenge_slug).list_training_datasets()

    @mcp.tool(annotations=READ_ONLY)
    def get_training_dataset_info(
        version: str, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Get details for one training dataset version.

        Args:
            version: A dataset version such as '1.0', or 'latest'.
        """
        return client_for(challenge_slug).get_training_dataset(version)

    @mcp.tool(annotations=READ_ONLY)
    def get_inference_data_info(
        release_date: str, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Get details for one inference period ('current' works).

        Args:
            release_date: YYYY-MM-DD, or 'current' for the active period.
        """
        return client_for(challenge_slug).get_inference_data(release_date)

    @mcp.tool(annotations=READ_ONLY)
    def get_training_dataset_url(
        version: str, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> str:
        """Get a signed, time-limited download URL for a training dataset.

        Works everywhere (no local filesystem needed): fetch the URL with
        any HTTP client.

        Args:
            version: A dataset version such as '1.0', or 'latest'.
        """
        return client_for(challenge_slug).get_training_dataset_url(version)

    @mcp.tool(annotations=READ_ONLY)
    def get_inference_data_url(
        release_date: str = "current", challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> str:
        """Get a signed, time-limited download URL for inference features.

        Args:
            release_date: YYYY-MM-DD, or 'current' (default).
        """
        return client_for(challenge_slug).get_inference_data_url(release_date)

    @mcp.tool(annotations=READ_ONLY)
    def get_meta_model_url(challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE) -> str:
        """Get a signed, time-limited download URL for the consolidated
        meta-model file."""
        return client_for(challenge_slug).get_meta_model_url()

    @mcp.tool
    def submit_predictions_from_dataframe(
        df: str, slot: int = 1, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Submit predictions from a JSON representation of a dataframe.

        The dataframe must contain the challenge's required columns (for
        hyperliquid-ranking: id, pred_10d, pred_30d).

        Args:
            df: JSON string in the {"column": [values, ...]} orientation.
            slot: Submission slot 1-5.
            challenge_slug: The challenge to submit to.
        """
        import narwhals as nw

        try:
            data = json.loads(df)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON dataframe: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("JSON dataframe must be an object of column arrays.")
        frame = nw.from_dict(data, backend="pyarrow")
        return client_for(challenge_slug).submit_predictions(df=frame, slot=slot)

    @mcp.tool(annotations=READ_ONLY)
    def list_submissions(
        period: Optional[str] = None, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> List[Dict[str, Any]]:
        """List your prediction submissions for a challenge.

        Args:
            period: Optional filter: 'current' or a YYYY-MM-DD release date.
        """
        return client_for(challenge_slug).list_submissions(period=period)

    @mcp.tool(annotations=READ_ONLY)
    def get_submission(
        submission_id: int, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Get details for one of your submissions by id.

        Args:
            submission_id: The submission id from list_submissions.
        """
        return client_for(challenge_slug).get_submission(submission_id)

    @mcp.tool(annotations=READ_ONLY)
    def get_performance(
        user: Optional[str] = None,
        scored_only: bool = True,
        slot: Optional[int] = None,
        daily: bool = False,
        challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE,
    ) -> List[Dict[str, Any]]:
        """Get a member's submission performance, newest first. Every member's
        scores are public. Percentiles run 0 to 100.

        Args:
            user: The member's username. Leave empty for your own.
            scored_only: Only include submissions that have been scored.
            slot: Optional slot filter (1-5).
            daily: Instead, return one row per metric and day for the member's
                pending submissions (the ones still being scored). `is_final`
                marks the day that becomes the final score.
        """
        return client_for(challenge_slug).get_performance(
            user=user,
            scored_only=scored_only,
            slot=slot,
            daily=daily,
            status="pending" if daily else None,
        )

    # Local-filesystem tools are meaningful only under stdio; on a shared
    # hosted server they would write to the service's own disk.
    if is_hosted():
        return

    @mcp.tool
    def download_training_dataset(
        version: str, dest_path: str, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> str:
        """Download a training dataset to a local file, parquet or CSV by extension.

        Args:
            version: Dataset version, or 'latest'.
            dest_path: Local path to write (.parquet appended if missing).
        """
        path = _coerced_path(dest_path)
        client_for(challenge_slug).download_training_dataset(version, str(path))
        return f"Training data downloaded to {path}"

    @mcp.tool
    def download_inference_data(
        release_date: str,
        dest_path: str,
        poll: bool = True,
        poll_interval: int = 30,
        timeout: Optional[int] = 900,
        challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE,
    ) -> str:
        """Download inference features to a local file, parquet or CSV by extension.

        Args:
            release_date: YYYY-MM-DD, or 'current'.
            dest_path: Local path to write (.parquet appended if missing).
            poll: Wait for the period's features if not yet available.
            poll_interval: Seconds between polls.
            timeout: Give up after this many seconds (None = wait forever).
        """
        path = _coerced_path(dest_path)
        client_for(challenge_slug).download_inference_data(
            release_date,
            str(path),
            poll=poll,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        return f"Inference data downloaded to {path}"

    @mcp.tool
    def download_meta_model(
        dest_path: str, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> str:
        """Download the consolidated meta-model to a local file, parquet or CSV by extension."""
        path = _coerced_path(dest_path)
        client_for(challenge_slug).download_meta_model(str(path))
        return f"Meta-model downloaded to {path}"

    @mcp.tool
    def submit_predictions_from_file(
        file_path: str, slot: int = 1, challenge_slug: ChallengeSlug = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Submit predictions from a local parquet or CSV file.

        Args:
            file_path: Path to a parquet or CSV file with the required columns.
            slot: Submission slot 1-5.
        """
        return client_for(challenge_slug).submit_predictions(
            file_path=file_path, slot=slot
        )
