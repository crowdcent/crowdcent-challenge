"""Client plumbing: auth/session setup, the request/download/redirect
helpers, and account-level calls shared by every API area."""

import hashlib
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, IO, List, Optional

import requests
from dotenv import load_dotenv
from requests import exceptions as requests_exceptions

from ..exceptions import (
    AuthenticationError,
    ClientError,
    CrowdCentAPIError,
    NotFoundError,
    ServerError,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _progress_bar(total_size: int, dest_path: str):
    """A tqdm bar that also works in sandboxed interpreters (Pyodide/WASM
    notebooks), where multiprocessing locks are unavailable and tqdm's default
    write lock raises while being built.

    The lock is probed through the public ``get_lock`` classmethod *before*
    any bar is constructed: if tqdm's default lock cannot be created, a plain
    threading lock is installed via ``set_lock``. Letting the constructor fail
    instead would leave a half-initialised instance behind, and its ``__del__``
    then prints an ``AttributeError`` traceback to stderr on collection."""
    from tqdm import tqdm

    try:
        tqdm.get_lock()
    except RuntimeError:
        tqdm.set_lock(threading.RLock())
    return tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(dest_path)}",
    )


class BaseClient:
    DEFAULT_BASE_URL = "https://crowdcent.com/api"
    DEFAULT_CHALLENGE_SLUG = "hyperliquid-ranking"
    API_KEY_ENV_VAR = "CROWDCENT_API_KEY"
    BASE_URL_ENV_VAR = "CROWDCENT_API_URL"

    def __init__(
        self,
        challenge_slug: str = DEFAULT_CHALLENGE_SLUG,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initializes the ChallengeClient for a specific challenge.

        Args:
            challenge_slug: The unique identifier (slug) for the challenge.
                            Defaults to "hyperliquid-ranking".
            api_key: Your CrowdCent API key. If not provided, it will attempt
                     to load from the CROWDCENT_API_KEY environment variable
                     or a .env file.
            base_url: The base URL of the CrowdCent API. If not provided, it
                      is read from the CROWDCENT_API_URL environment variable,
                      and defaults to https://crowdcent.com/api.
        """
        load_dotenv()  # Load .env file if present
        self.api_key = api_key or os.getenv(self.API_KEY_ENV_VAR)
        if not self.api_key:
            raise AuthenticationError(
                f"API key not provided and not found in environment variable "
                f"'{self.API_KEY_ENV_VAR}' or .env file."
            )

        self.challenge_slug = challenge_slug
        self.base_url = (
            base_url or os.getenv(self.BASE_URL_ENV_VAR) or self.DEFAULT_BASE_URL
        ).rstrip("/")
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
        allow_redirects: bool = True,
        headers: Optional[Dict[str, str]] = None,
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
            headers: Extra per-request headers (e.g. Idempotency-Key), merged
                over the session's auth header by requests.

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
                    allow_redirects=allow_redirects,
                    headers=headers,
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

    def _download_file(self, endpoint: str, dest_path: str, description: str, *, params=None) -> None:
        """Download a file from the API with progress bar.

        Args:
            endpoint: API endpoint to download from.
            dest_path: Local file path to save to. A ``.csv`` path asks the API
                for the CSV copy of the file; anything else is parquet.
            description: Human-readable description for logging (e.g., "training data v1.0").
        """
        logger.info(f"Downloading {description} to {dest_path}")
        if params is None:
            params = {"as": "csv"} if str(dest_path).lower().endswith(".csv") else None
        response = self._request("GET", endpoint, params=params, stream=True)
        temporary = None
        try:
            total_size = int(response.headers.get("content-length", 0))
            digest = hashlib.sha256()
            with (
                tempfile.NamedTemporaryFile(
                    mode="wb", dir=os.path.dirname(os.path.abspath(dest_path)),
                    prefix=".crowdcent-download-", delete=False,
                ) as f,
                _progress_bar(total_size, dest_path) as pbar,
            ):
                temporary = f.name
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    digest.update(chunk)
                    pbar.update(len(chunk))
            expected = (params or {}).get("sha256")
            if expected and digest.hexdigest() != expected:
                raise CrowdCentAPIError("Downloaded file does not match the requested SHA-256.")
            os.replace(temporary, dest_path)
            logger.info(f"Successfully downloaded {description} to {dest_path}")
        except IOError as e:
            raise CrowdCentAPIError(f"Could not download file: {e}") from e
        finally:
            response.close()
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

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
            base_url: The base URL of the CrowdCent API. If not provided, it
                      is read from the CROWDCENT_API_URL environment variable,
                      and defaults to https://crowdcent.com/api.

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

        base_url = (
            base_url or os.getenv(cls.BASE_URL_ENV_VAR) or cls.DEFAULT_BASE_URL
        ).rstrip("/")
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

    # --- Auth ---

    def check_auth(self) -> Dict[str, Any]:
        """Checks the presenting API key and reports its capabilities.

        A cheap validity probe: use it to fail fast at startup, or to decide
        whether trading features should be surfaced at all.

        Returns:
            A dictionary with:
                - `username`: The account the key belongs to.
                - `allow_trading`: Whether this key may call mutating trading
                  endpoints (the per-key "Allow live trading" switch).
                - `oms_access`: Whether the trading API is open to this user
                  at all (Challenger tier with active submission).
                - `allow_cloud`: Whether this key may use CrowdCent Cloud
                  (the per-key "Allow Cloud" switch). Calls also require
                  account eligibility; public preview requires Challenger+.

        Raises:
            AuthenticationError: If the key is invalid or revoked.
        """
        response = self._request("GET", "/auth/check/")
        return response.json()

    # --- Signed download URLs (hosted-agent friendly) ---

    def _signed_url(self, endpoint: str) -> str:
        """Resolve a download endpoint's redirect target without downloading."""
        response = self._request("GET", endpoint, allow_redirects=False)
        url = response.headers.get("Location")
        if not url:
            raise CrowdCentAPIError(
                f"Expected a redirect from {endpoint}, got {response.status_code}."
            )
        return url
