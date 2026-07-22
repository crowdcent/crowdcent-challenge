"""Prediction submissions and scored performance."""

import logging
import os
from typing import Any, Dict, List, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT  # noqa: F401

from ..exceptions import ClientError, CrowdCentAPIError  # noqa: F401

logger = logging.getLogger(__name__)


class SubmissionsAPI:
    # --- Submission Methods ---

    def list_submissions(self, period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists the authenticated user's submissions for this challenge.

        Args:
            period: Optional filter for submissions by period:
                  - 'current': Only show submissions for the current active period
                  - 'YYYY-MM-DD': Only show submissions for a specific inference period date

        Returns:
            A list of dictionaries, each representing a submission.
        """
        params = {}
        if period:
            params["period"] = period

        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/submissions/", params=params
        )
        return response.json()

    def get_submission(self, submission_id: int) -> Dict[str, Any]:
        """Gets details for a specific submission by its ID.

        Args:
            submission_id: The ID of the submission to retrieve.

        Returns:
            A dictionary representing the specified submission.

        Raises:
            NotFoundError: If the submission with the given ID is not found
                           or doesn't belong to the user.
        """
        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/submissions/{submission_id}/"
        )
        return response.json()

    @nw.narwhalify
    def submit_predictions(
        self,
        file_path: str = "submission.parquet",
        df: Optional[IntoFrameT] = None,
        slot: int = 1,
        queue_next: bool = True,
        temp: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        is_experimental: bool = False,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Submit predictions for this challenge.

        If a submission window is currently open, the prediction is submitted immediately.
        If no window is open, the prediction is queued and will be automatically submitted
        when the next window opens.

        You can provide either a file path to an existing Parquet file or a DataFrame
        that will be temporarily saved as Parquet for submission.

        The data must contain the required prediction columns specified by the challenge
        (e.g., id, pred_10d, pred_30d).

        Args:
            file_path: Optional path to an existing prediction Parquet file.
            df: Optional DataFrame with the prediction columns. If provided,
                it will be temporarily saved as Parquet for submission.
            slot: Submission slot number (1-based).
            queue_next: Whether to also queue this submission for the next period
                (auto-rollover). Defaults to True. When submitting during an open
                window, this queues a copy for the following period.
            temp: Whether to save the DataFrame to a temporary file.
            max_retries: Maximum number of retry attempts for connection errors (default: 3).
            retry_delay: Initial delay between retries in seconds (default: 1.0).
            is_experimental: Mark this submission as experimental. Experimental
                submissions are scored normally and receive a shadow percentile
                against the non-experimental competitive field, but are excluded
                from the leaderboard, the meta-model, and CC Points performance
                adjustment. **Constraint:** at least one slot per period must
                have a non-experimental submission; submitting experimental
                without a non-experimental sibling in another slot is rejected.
                Defaults to False.
            notes: Free-text annotation for this submission, max 2000 characters.
                Notes are private to the submission owner. Defaults to "".

        Returns:
            A dictionary with submission details. The shape depends on context:

            - **Window open (immediate submission)**: Contains submission fields
                like `id`, `status`, `slot`, `submitted_at`, `is_experimental`,
                `notes`, plus `queued_for_next` (bool). If the queue copy was
                rejected, `queue_error_code` and `queue_error` are populated.
            - **Window closed (queued)**: Contains `status: "queued"`, `slot`,
                `challenge`, `is_experimental`, `notes`, and a `message`
                describing when it will be submitted.

        Raises:
            ValueError: If neither file_path nor df is provided, or if both are provided.
            FileNotFoundError: If the specified file_path does not exist.
            ClientError: If the submission is invalid (e.g., wrong format, missing columns,
                experimental constraint violated).

        Examples:
            # Submit from a DataFrame
            client.submit_predictions(df=predictions_df)

            # Submit from a file
            client.submit_predictions(file_path="predictions.parquet")

            # Submit and opt-out of auto-queueing for next period
            client.submit_predictions(df=predictions_df, queue_next=False)

            # Submit an experimental prediction with a note
            client.submit_predictions(
                df=predictions_df,
                slot=2,
                is_experimental=True,
                notes="2-layer transformer w/ sector embeddings",
            )
        """
        if df is not None:
            df.write_parquet(file_path)
            logger.info(f"Wrote DataFrame to temporary file: {file_path}")

        logger.info(
            f"Submitting predictions from {file_path} to challenge '{self.challenge_slug}' (Slot: {slot or '1'})"
        )

        try:
            with open(file_path, "rb") as f:
                files = {
                    "prediction_file": (
                        os.path.basename(file_path),
                        f,
                        "application/octet-stream",
                    )
                }
                data_payload = {
                    "slot": str(slot),
                    "also_queue_next": str(queue_next).lower(),
                    "is_experimental": str(is_experimental).lower(),
                    "notes": notes,
                }
                response = self._request(
                    "POST",
                    f"/challenges/{self.challenge_slug}/submissions/",
                    files=files,
                    data=data_payload,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )

            resp_data = response.json()

            # 202=queued, 200=updated, 201=created
            msg = {202: "queued", 200: "updated", 201: "created"}.get(
                response.status_code, "submitted"
            )
            exp_label = " (experimental)" if is_experimental else ""
            logger.info(f"Submission {msg} (slot {slot}){exp_label}")
            if resp_data.get("queued_for_next"):
                logger.info("Also queued for next period.")
            elif resp_data.get("queue_error_code"):
                logger.warning(
                    "Live submission saved but queue copy was rejected: "
                    f"{resp_data['queue_error_code']} - {resp_data.get('queue_error')}"
                )

            return resp_data
        except FileNotFoundError as e:
            logger.error(f"Prediction file not found at {file_path}")
            raise FileNotFoundError(f"Prediction file not found at {file_path}") from e
        except IOError as e:
            logger.error(f"Failed to read prediction file {file_path}: {e}")
            raise CrowdCentAPIError(f"Failed to read prediction file: {e}") from e
        finally:
            # Clean up the temporary file if we created one
            if df is not None and temp:
                try:
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up temporary file {file_path}: {e}"
                    )

    # --- Historical Performance Methods ---

    def get_performance(
        self,
        user: Optional[str] = None,
        scored_only: bool = True,
        slot: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get performance history for a user (defaults to authenticated user).

        Fetches submissions with their scores and percentiles, flattens the
        nested score data, and returns a list ready to wrap in pandas/polars.

        Args:
            user: Username to fetch performance for. If None (default), fetches
                performance for the authenticated user.
                *Note: Fetching other users' performance is not yet supported.*
            scored_only: If True (default), only include submissions that have been scored.
                For pending submissions, only the most recent (partially resolved) score
                is available — daily granularity is not currently exposed by the API.
            slot: Optional slot filter. If provided, only include submissions from this slot.

        Returns:
            A list of dictionaries, each containing:
            - id: Submission ID
            - slot: Submission slot number
            - release_date: The inference period date (ISO string)
            - submitted_at: When the submission was made (ISO string)
            - status: Submission status ("pending" or "evaluated")
            - is_experimental: Whether this submission was marked experimental (bool)
            - notes: Free-text note attached at submit time (str, may be empty)
            - score_*: Individual score metrics (e.g., score_spearman_10d)
            - percentile_*: Individual percentile metrics (e.g., percentile_spearman_10d)
            - composite_percentile: Overall percentile ranking (if available)

        Note:
            - For submissions with status="pending", scores reflect the most recent
              partial evaluation (e.g., day 5 of a 10-day prediction). Daily score
              progression is tracked server-side but not yet available via the API.

            - Percentile fields (e.g., composite_percentile=0.75) indicate rank
              relative to all participants — 0.75 means outperforming 75% of
              submissions for that period.

            - Experimental submissions are included in the result. They are
              scored against the non-experimental competitive field (shadow
              percentiles) but excluded from leaderboards, the meta-model, and
              CC Points performance adjustment. Filter them out client-side if
              you want only competitive history:
              ``[r for r in rows if not r["is_experimental"]]``.

        Example:
            >>> client = ChallengeClient("momentum-alpha")
            >>> history = client.get_performance()
            >>> import pandas as pd
            >>> df = pd.DataFrame(history)
        """
        if user is not None:
            raise NotImplementedError(
                "Fetching performance for specific users is not yet supported via the API. "
                "Leave `user=None` to fetch your own performance."
            )

        logger.info(f"Fetching submission history for '{self.challenge_slug}'...")
        submissions = self.list_submissions()

        if not submissions:
            logger.info("No submissions found.")
            return []

        rows = []
        for sub in submissions:
            # Skip unscored if requested
            if scored_only and not sub.get("score_details"):
                continue

            # Skip if slot filter doesn't match
            if slot is not None and sub.get("slot") != slot:
                continue

            row = {
                "id": sub.get("id"),
                "slot": sub.get("slot"),
                "release_date": sub.get("inference_data_release_date", "")[:10]
                if sub.get("inference_data_release_date")
                else None,
                "submitted_at": sub.get("submitted_at"),
                "status": sub.get("status"),
                "is_experimental": sub.get("is_experimental", False),
                "notes": sub.get("notes", "") or "",
            }

            # Flatten score_details (avoid redundant prefix if key already contains it)
            score_details = sub.get("score_details") or {}
            for key, value in score_details.items():
                col = key if "score" in key else f"score_{key}"
                row[col] = value

            # Flatten percentile_details (avoid redundant prefix if key already contains it)
            percentile_details = sub.get("percentile_details") or {}
            for key, value in percentile_details.items():
                col = key if "percentile" in key else f"percentile_{key}"
                row[col] = value

            rows.append(row)

        # Sort by release_date descending (most recent first)
        rows.sort(key=lambda x: x.get("release_date") or "", reverse=True)

        logger.info(f"Loaded {len(rows)} scored submissions.")
        return rows
