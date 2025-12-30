import os
import requests
from requests import exceptions as requests_exceptions
from dotenv import load_dotenv
from typing import Optional, Dict, Any, IO, List
import logging
from datetime import datetime
import narwhals as nw
from narwhals.typing import IntoFrameT
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class CrowdCentAPIError(Exception):
    """Base exception for API errors."""

    pass


class AuthenticationError(CrowdCentAPIError):
    """Exception for authentication issues."""

    pass


class NotFoundError(CrowdCentAPIError):
    """Exception for 404 errors."""

    pass


class ClientError(CrowdCentAPIError):
    """Exception for 4xx client errors (excluding 401, 404)."""

    pass


class ServerError(CrowdCentAPIError):
    """Exception for 5xx server errors."""

    pass


class ChallengeClient:
    """
    Client for interacting with a specific CrowdCent Challenge.

    Handles authentication and provides methods for accessing challenge data,
    training datasets, inference data, and managing prediction submissions for
    a specific challenge identified by its slug.
    """

    DEFAULT_BASE_URL = "https://crowdcent.com/api"
    API_KEY_ENV_VAR = "CROWDCENT_API_KEY"

    def __init__(
        self,
        challenge_slug: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initializes the ChallengeClient for a specific challenge.

        Args:
            challenge_slug: The unique identifier (slug) for the challenge.
            api_key: Your CrowdCent API key. If not provided, it will attempt
                     to load from the CROWDCENT_API_KEY environment variable
                     or a .env file.
            base_url: The base URL of the CrowdCent API. Defaults to
                      https://crowdcent.com/api.
        """
        load_dotenv()  # Load .env file if present
        self.api_key = api_key or os.getenv(self.API_KEY_ENV_VAR)
        if not self.api_key:
            raise AuthenticationError(
                f"API key not provided and not found in environment variable "
                f"'{self.API_KEY_ENV_VAR}' or .env file."
            )

        self.challenge_slug = challenge_slug
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Api-Key {self.api_key}"})
        logger.info(
            f"ChallengeClient initialized for '{challenge_slug}' at URL: {self.base_url}"
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        files: Optional[Dict[str, IO]] = None,
        stream: bool = False,
        data: Optional[Dict] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> requests.Response:
        """
        Internal helper method to make authenticated API requests.

        Args:
            method: HTTP method (e.g., 'GET', 'POST').
            endpoint: API endpoint path (e.g., '/challenges/').
            params: URL parameters.
            json_data: JSON data for the request body.
            files: Files to upload (for multipart/form-data).
            stream: Whether to stream the response (for downloads).
            data: Dictionary of form data to send with multipart requests.
            max_retries: Maximum number of retry attempts for connection errors.
            retry_delay: Initial delay between retries (seconds). Will use exponential backoff.

        Returns:
            The requests.Response object.

        Raises:
            AuthenticationError: If the API key is invalid (401).
            NotFoundError: If the resource is not found (404).
            ClientError: For other 4xx errors.
            ServerError: For 5xx errors.
            CrowdCentAPIError: For other request exceptions.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(
            f"Request: {method} {url} Params: {params} JSON: {json_data is not None} "
            f"Data: {data is not None} Files: {files is not None}"
        )

        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                    files=files,
                    stream=stream,
                    data=data,
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                logger.debug(f"Response: {response.status_code}")
                return response
            except requests_exceptions.HTTPError as e:
                status_code = e.response.status_code

                # Try to parse standardized error format: {"error": {"code": "ERROR_CODE", "message": "Description"}}
                try:
                    error_data = e.response.json()
                    if "error" in error_data and isinstance(error_data["error"], dict):
                        error_code = error_data["error"].get("code", "UNKNOWN_ERROR")
                        error_message = error_data["error"].get(
                            "message", e.response.text
                        )
                    else:
                        error_code = "API_ERROR"
                        error_message = e.response.text
                except requests_exceptions.JSONDecodeError:
                    error_code = "API_ERROR"
                    error_message = e.response.text

                logger.error(
                    f"API Error ({status_code}): {error_code} - {error_message} for {method} {url}"
                )

                if status_code == 401:
                    raise AuthenticationError(
                        f"Authentication failed (401): {error_message} [{error_code}]"
                    ) from e
                elif status_code == 404:
                    raise NotFoundError(
                        f"Resource not found (404): {error_message} [{error_code}]"
                    ) from e
                elif 400 <= status_code < 500:
                    raise ClientError(
                        f"Client error ({status_code}): {error_message} [{error_code}]"
                    ) from e
                elif 500 <= status_code < 600:
                    raise ServerError(
                        f"Server error ({status_code}): {error_message} [{error_code}]"
                    ) from e
                else:
                    raise CrowdCentAPIError(
                        f"HTTP error ({status_code}): {error_message} [{error_code}]"
                    ) from e
            except (
                requests_exceptions.ConnectionError,
                requests_exceptions.Timeout,
            ) as e:
                # Connection errors and timeouts are retryable
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Connection error: {e}. Retrying in {delay:.1f}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                logger.error(
                    f"Request failed after {max_retries} retries: {e} for {method} {url}"
                )
                raise CrowdCentAPIError(
                    f"Request failed after {max_retries} retries: {e}"
                ) from e
            except requests_exceptions.RequestException as e:
                logger.error(f"Request failed: {e} for {method} {url}")
                raise CrowdCentAPIError(f"Request failed: {e}") from e

    def _download_file(self, endpoint: str, dest_path: str, description: str) -> None:
        """Download a file from the API with progress bar.

        Args:
            endpoint: API endpoint to download from.
            dest_path: Local file path to save to.
            description: Human-readable description for logging (e.g., "training data v1.0").
        """
        logger.info(f"Downloading {description} to {dest_path}")
        response = self._request("GET", endpoint, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        try:
            from tqdm import tqdm

            with open(dest_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(dest_path)}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            logger.info(f"Successfully downloaded {description} to {dest_path}")
        except IOError as e:
            logger.error(f"Failed to write to {dest_path}: {e}")
            raise CrowdCentAPIError(f"Failed to write file: {e}") from e

    # --- Class Method for Listing All Challenges ---

    @classmethod
    def list_all_challenges(
        cls, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists all active challenges.

        This is a class method that doesn't require a challenge_slug.
        Use this to discover available challenges before initializing a ChallengeClient.

        Args:
            api_key: Your CrowdCent API key. If not provided, it will attempt
                     to load from the CROWDCENT_API_KEY environment variable
                     or a .env file.
            base_url: The base URL of the CrowdCent API. Defaults to
                      http://crowdcent.com/api.

        Returns:
            A list of dictionaries, each representing an active challenge.
        """
        # Create a temporary session for this request
        load_dotenv()
        api_key = api_key or os.getenv(cls.API_KEY_ENV_VAR)
        if not api_key:
            raise AuthenticationError(
                f"API key not provided and not found in environment variable "
                f"'{cls.API_KEY_ENV_VAR}' or .env file."
            )

        base_url = (base_url or cls.DEFAULT_BASE_URL).rstrip("/")
        session = requests.Session()
        session.headers.update({"Authorization": f"Api-Key {api_key}"})

        url = f"{base_url}/challenges/"
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.json()
        except requests_exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 401:
                raise AuthenticationError("Authentication failed (401)")
            elif status_code == 404:
                raise NotFoundError("Resource not found (404)")
            elif 400 <= status_code < 500:
                raise ClientError(f"Client error ({status_code})")
            elif 500 <= status_code < 600:
                raise ServerError(f"Server error ({status_code})")
            else:
                raise CrowdCentAPIError(f"HTTP error ({status_code})")
        except requests_exceptions.RequestException as e:
            raise CrowdCentAPIError(f"Request failed: {e}")

    # --- Challenge Methods ---

    def get_challenge(self) -> Dict[str, Any]:
        """Gets details for this challenge.

        Returns:
            A dictionary representing this challenge.

        Raises:
            NotFoundError: If the challenge with the given slug is not found.
        """
        response = self._request("GET", f"/challenges/{self.challenge_slug}/")
        return response.json()

    # --- Training Data Methods ---

    def list_training_datasets(self) -> List[Dict[str, Any]]:
        """Lists all training dataset versions for this challenge.

        Returns:
            A list of dictionaries, each representing a training dataset version.

        Raises:
            NotFoundError: If the challenge is not found.
        """
        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/training_data/"
        )
        return response.json()

    def get_training_dataset(self, version: str) -> Dict[str, Any]:
        """Gets details for a specific training dataset version.

        Args:
            version: The version string of the training dataset (e.g., '1.0', '2.1')
                     or the special value ``"latest"`` to get the latest version.

        Returns:
            A dictionary representing the specified training dataset.

        Raises:
            NotFoundError: If the challenge or the specified training dataset is not found.
        """
        if version == "latest":
            response = self._request(
                "GET", f"/challenges/{self.challenge_slug}/training_data/latest/"
            )
            return response.json()

        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/training_data/{version}/"
        )
        return response.json()

    def download_training_dataset(self, version: str, dest_path: str):
        """Downloads the training data file for a specific dataset version.

        Args:
            version: The version string of the training dataset (e.g., '1.0', '2.1')
                    or 'latest' to get the latest version.
            dest_path: The local file path to save the downloaded dataset.

        Raises:
            NotFoundError: If the challenge, dataset, or its file is not found.
        """
        if version == "latest":
            latest_info = self.get_training_dataset("latest")
            version = latest_info["version"]

        endpoint = (
            f"/challenges/{self.challenge_slug}/training_data/{version}/download/"
        )
        self._download_file(endpoint, dest_path, f"training data v{version}")

    # --- Inference Data Methods ---

    def list_inference_data(self) -> List[Dict[str, Any]]:
        """Lists all inference data periods for this challenge.

        Returns:
            A list of dictionaries, each representing an inference data period.

        Raises:
            NotFoundError: If the challenge is not found.
        """
        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/inference_data/"
        )
        return response.json()

    def get_inference_data(self, release_date: str) -> Dict[str, Any]:
        """Gets details for a specific inference data period by its release date.

        Args:
            release_date: The release date of the inference data in 'YYYY-MM-DD' format.
                          You can also pass the special values:
                          - ``"current"`` to fetch the current active inference period
                          - ``"latest"`` to fetch the most recently *available* inference period

        Returns:
            A dictionary representing the specified inference data period.

        Raises:
            NotFoundError: If the challenge or the specified inference data is not found.
            ClientError: If the date format is invalid.
        """
        if release_date == "current":
            response = self._request(
                "GET", f"/challenges/{self.challenge_slug}/inference_data/current/"
            )
            return response.json()

        if release_date == "latest":
            # Simply resolve via list_inference_data(); avoid noisy probe.
            periods = self.list_inference_data()
            if not periods:
                raise NotFoundError(
                    "No inference data periods found for this challenge."
                )

            latest_period = max(periods, key=lambda p: p["release_date"])
            release_date_iso = latest_period["release_date"]
            release_date = release_date_iso.split("T")[0]

        # Validate date format for explicit dates
        try:
            datetime.strptime(release_date, "%Y-%m-%d")
        except ValueError:
            raise ClientError(
                f"Invalid date format: {release_date}. Use 'YYYY-MM-DD' format."
            )

        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/inference_data/{release_date}/"
        )
        return response.json()

    def download_inference_data(
        self,
        release_date: str,
        dest_path: str,
        poll: bool = True,
        poll_interval: int = 30,
        timeout: Optional[int] = 900,
    ):
        """Downloads the inference features file for a specific period.

        Args:
            release_date: The release date of the inference data in 'YYYY-MM-DD' format
                          or the special values ``"current"`` or ``"latest"``.
            dest_path: The local file path to save the downloaded features file.
            poll: Whether to wait for the inference data to be available before downloading.
            poll_interval: Seconds to wait between retries when polling.
            timeout: Maximum seconds to wait before raising :class:`TimeoutError`.
                ``None`` waits indefinitely.

        Raises:
            NotFoundError: If the challenge, inference data, or its file is not found.
            ClientError: If the date format is invalid.
        """
        if release_date == "current":
            # If polling is enabled, delegate to wait_for_inference_data which wraps
            # this method and adds retry logic. Otherwise attempt a single direct
            # download request.
            if poll:
                self.wait_for_inference_data(dest_path, poll_interval, timeout)
                return

            # Polling disabled → attempt once and propagate NotFoundError on 404.
            endpoint = (
                f"/challenges/{self.challenge_slug}/inference_data/current/download/"
            )
        else:
            if release_date == "latest":
                latest_info = self.get_inference_data("latest")
                release_date_iso = latest_info.get("release_date")
                release_date = (
                    release_date_iso.split("T")[0] if release_date_iso else None
                )
                if not release_date:
                    raise CrowdCentAPIError(
                        "Malformed response when resolving latest inference period."
                    )

            # Validate date format after any resolution.
            try:
                datetime.strptime(release_date, "%Y-%m-%d")
            except ValueError:
                raise ClientError(
                    f"Invalid date format: {release_date}. Use 'YYYY-MM-DD' format."
                )

            endpoint = f"/challenges/{self.challenge_slug}/inference_data/{release_date}/download/"

        self._download_file(endpoint, dest_path, f"inference data {release_date}")

    def wait_for_inference_data(
        self,
        dest_path: str,
        poll_interval: int = 30,
        timeout: Optional[int] = 900,
    ) -> None:
        """Waits for the *current* inference data release to appear and downloads it.

        The internal data-generation pipeline begins around 14:00 UTC, but the
        public inference file becomes available only after it passes data-quality
        checks. This helper repeatedly calls
        :py:meth:`download_inference_data` with ``release_date="current"`` until
        the file is ready (HTTP 404s are silently retried).

        Args:
            dest_path: Local path where the parquet file will be saved once available.
            poll_interval: Seconds to wait between retries.
            timeout: Maximum seconds to wait before raising :class:`TimeoutError`.
                ``None`` waits indefinitely.

        Raises:
            TimeoutError: If *timeout* seconds pass without a successful download.
            CrowdCentAPIError: For unrecoverable errors returned by the API.
        """
        start_time = time.time()
        attempts = 0

        while True:
            attempts += 1
            try:
                # Try to download the *current* period *once*. Pass poll=False to avoid
                # the mutual recursion between `wait_for_inference_data` and
                # `download_inference_data` which would otherwise trigger an infinite
                # loop when the file is not yet available.
                self.download_inference_data("current", dest_path, poll=False)
                logger.info(
                    f"Successfully downloaded inference data after {attempts} attempt(s) to {dest_path}"
                )
                return  # Success – exit the loop
            except NotFoundError:
                # File not published yet – check timeout and sleep before retrying.
                elapsed = time.time() - start_time
                if timeout is not None and elapsed >= timeout:
                    raise TimeoutError(
                        f"Inference data was not available after waiting {timeout} seconds."
                    )
                logger.debug(
                    f"Inference data not yet available (attempt {attempts}). "
                    f"Sleeping {poll_interval}s before retrying."
                )
                time.sleep(poll_interval)

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

        Returns:
            A dictionary with submission details. The shape depends on context:

            - **Window open (immediate submission)**: Contains submission fields like
                `id`, `status`, `slot`, `submitted_at`, plus `queued_for_next` (bool).
            - **Window closed (queued)**: Contains `status: "queued"`, `slot`,
                `challenge`, and a `message` describing when it will be submitted.

        Raises:
            ValueError: If neither file_path nor df is provided, or if both are provided.
            FileNotFoundError: If the specified file_path does not exist.
            ClientError: If the submission is invalid (e.g., wrong format, missing columns).

        Examples:
            # Submit from a DataFrame
            client.submit_predictions(df=predictions_df)

            # Submit from a file
            client.submit_predictions(file_path="predictions.parquet")

            # Submit and opt-out of auto-queueing for next period
            client.submit_predictions(df=predictions_df, queue_next=False)
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
                }
                response = self._request(
                    "POST",
                    f"/challenges/{self.challenge_slug}/submissions/",
                    files=files,
                    data=data_payload,  # Pass slot and queue flag in data
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )

            resp_data = response.json()

            # 202=queued, 200=updated, 201=created
            msg = {202: "queued", 200: "updated", 201: "created"}.get(
                response.status_code, "submitted"
            )
            logger.info(f"Submission {msg} (slot {slot})")
            if resp_data.get("queued_for_next"):
                logger.info("Also queued for next period.")

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

    # --- Challenge Switching ---

    def switch_challenge(self, new_challenge_slug: str) -> None:
        """Switch this client to interact with a different challenge.

        Args:
            new_challenge_slug: The slug identifier for the new challenge.

        Returns:
            None. The client is modified in-place.
        """
        self.challenge_slug = new_challenge_slug
        logger.info(f"Client switched to challenge '{new_challenge_slug}'")

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

    # --- Meta-Model Download ---

    def download_meta_model(self, dest_path: str):
        """Downloads the consolidated meta-model file for this challenge.

        The meta-model is typically an aggregation (e.g., average) of all valid
        submissions for past inference periods.

        Args:
            dest_path: The local file path to save the downloaded meta-model.

        Raises:
            NotFoundError: If the challenge or its meta-model file is not found.
            CrowdCentAPIError: For issues during download or file writing.
            PermissionDenied: If the meta-model is not public and user lacks permission.
        """
        endpoint = f"/challenges/{self.challenge_slug}/meta_model/download/"
        self._download_file(endpoint, dest_path, "meta-model")
