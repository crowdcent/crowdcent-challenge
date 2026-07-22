"""Challenge info, training/inference datasets, the meta-model, and
their signed-URL twins."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional  # noqa: F401

from ..exceptions import ClientError, CrowdCentAPIError, NotFoundError  # noqa: F401

logger = logging.getLogger(__name__)


class DataAPI:
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

    def get_training_dataset_url(self, version: str) -> str:
        """Returns a signed, time-limited download URL for a training dataset.

        The URL twin of :py:meth:`download_training_dataset` for callers
        without a local filesystem (hosted agents): fetch it yourself with
        any HTTP client.

        Args:
            version: The version string, or ``"latest"``.
        """
        if version == "latest":
            version = self.get_training_dataset("latest")["version"]
        return self._signed_url(
            f"/challenges/{self.challenge_slug}/training_data/{version}/download/"
        )

    def get_inference_data_url(self, release_date: str) -> str:
        """Returns a signed, time-limited download URL for inference features.

        Args:
            release_date: ``YYYY-MM-DD``, ``"current"``, or ``"latest"``.
        """
        if release_date == "latest":
            release_date = self.get_inference_data("latest")["release_date"].split("T")[
                0
            ]
        if release_date == "current":
            endpoint = (
                f"/challenges/{self.challenge_slug}/inference_data/current/download/"
            )
        else:
            endpoint = (
                f"/challenges/{self.challenge_slug}/inference_data/"
                f"{release_date}/download/"
            )
        return self._signed_url(endpoint)

    def get_meta_model_url(self) -> str:
        """Returns a signed, time-limited download URL for the consolidated
        meta-model file."""
        return self._signed_url(
            f"/challenges/{self.challenge_slug}/meta_model/download/"
        )
