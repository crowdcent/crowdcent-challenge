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
        file_path: str = "submission.csv",
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

        You can provide either a file path to an existing parquet or CSV file, or a
        DataFrame that will be temporarily saved for submission (as CSV, or as
        parquet when ``file_path`` ends in ``.parquet``).

        The data must contain the required prediction columns specified by the challenge
        (e.g., id, pred_10d, pred_30d).

        Args:
            file_path: Optional path to an existing prediction file, parquet or CSV.
            df: Optional DataFrame with the prediction columns. If provided,
                it is written to ``file_path`` for submission, in the format
                its extension names.
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

            # Submit from a file, parquet or CSV
            client.submit_predictions(file_path="predictions.parquet")
            client.submit_predictions(file_path="predictions.csv")

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
            if str(file_path).lower().endswith(".parquet"):
                df.write_parquet(file_path)
            else:
                df.write_csv(file_path)
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
        daily: bool = False,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get the performance history of any member (defaults to you).

        Every member's scores are public, as on their profile's Scores tab.
        Returns a list ready to wrap in pandas/polars, newest period first.

        Args:
            user: The member's username. If None (default), fetches your own.
            scored_only: If True (default), only include submissions that have been scored.
            slot: Optional slot filter. If provided, only include submissions from this slot.
            daily: If False (default), one row per submission with its scores.
                If True, one row per submission, metric and scored day.
            status: Optional status filter, "pending" or "evaluated".

        Returns:
            With ``daily=False``, one dictionary per submission:

            - id, slot, release_date (YYYY-MM-DD), submitted_at, status,
              is_experimental
            - notes: Free-text note attached at submit time (your own submissions only)
            - score_*: Each score (e.g., score_spearman_10d)
            - percentile_*: Each percentile, 0 to 100 (e.g., percentile_spearman_10d)
            - composite_percentile: Overall percentile, 0 to 100 (if available)

            With ``daily=True``, one dictionary per submission, metric and day:

            - id, slot, release_date, status, is_experimental
            - metric: e.g. "spearman_10d"
            - day: Days after release (1, 2, ...)
            - score: The metric scored against the return over the first ``day`` days
            - percentile: That day's percentile, 0 to 100 (None if not computed)
            - is_final: True on the day that is the metric's final score
              (day 10 for ``_10d`` metrics, day 30 for ``_30d`` metrics)

        Note:
            - A pending submission's scores are provisional. Its 10d scores are
              final after day 10 and its 30d scores after day 30.

            - A ``_10d`` metric keeps a daily score up to day 30, but only day 10
              counts as its final score.

            - Experimental submissions are included. They are scored against the
              non-experimental field (shadow percentiles) but excluded from
              leaderboards, the meta-model, and CC Points. Filter them out with
              ``[r for r in rows if not r["is_experimental"]]``.

        Example:
            >>> client = ChallengeClient("hyperliquid-ranking")
            >>> mine = client.get_performance()
            >>> alice = client.get_performance(user="alice", daily=True)
            >>> import polars as pl
            >>> df = pl.DataFrame(alice)
        """
        params: Dict[str, Any] = {}
        if user is not None:
            params["user"] = user
        if daily:
            params["daily"] = "true"
        if status is not None:
            params["status"] = status

        logger.info(
            f"Fetching submission history of {user or 'you'} for '{self.challenge_slug}'..."
        )
        submissions: List[Dict[str, Any]] = []
        page = 1
        while True:
            body = self._request(
                "GET",
                f"/challenges/{self.challenge_slug}/scores/",
                params={**params, "page": page},
            ).json()
            submissions.extend(body["results"])
            if not body.get("next"):
                break
            page += 1

        rows = []
        for sub in submissions:
            if scored_only and not sub.get("score_details"):
                continue
            if slot is not None and sub.get("slot") != slot:
                continue
            release = sub.get("inference_data_release_date")
            base = {
                "id": sub.get("id"),
                "slot": sub.get("slot"),
                "release_date": release[:10] if release else None,
                "status": sub.get("status"),
                "is_experimental": sub.get("is_experimental", False),
            }
            if daily:
                rows.extend(_daily_rows(base, sub))
                continue

            row = {
                "id": base["id"],
                "slot": base["slot"],
                "release_date": base["release_date"],
                "submitted_at": sub.get("submitted_at"),
                "status": base["status"],
                "is_experimental": base["is_experimental"],
            }
            if "notes" in sub:
                row["notes"] = sub["notes"] or ""
            # Flatten, without doubling a prefix the key already has.
            for key, value in (sub.get("score_details") or {}).items():
                row[key if "score" in key else f"score_{key}"] = value
            for key, value in (sub.get("percentile_details") or {}).items():
                row[key if "percentile" in key else f"percentile_{key}"] = value
            rows.append(row)

        logger.info(f"Loaded {len(rows)} rows.")
        return rows


_HORIZONS = {"_10d": 10, "_30d": 30}


def _daily_rows(base: Dict[str, Any], sub: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per metric and scored day, oldest day first."""
    percentiles = sub.get("daily_percentile_details") or {}
    rows = []
    for metric, days in (sub.get("daily_score_details") or {}).items():
        horizon = next((n for s, n in _HORIZONS.items() if metric.endswith(s)), None)
        for key in sorted(days, key=lambda k: int(k.removeprefix("day_"))):
            day = int(key.removeprefix("day_"))
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "day": day,
                    "score": days[key],
                    "percentile": percentiles.get(metric, {}).get(key),
                    "is_final": day == horizon,
                }
            )
    return rows
