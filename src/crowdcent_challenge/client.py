import os
import requests
from requests import exceptions as requests_exceptions
from dotenv import load_dotenv
from typing import Optional, Dict, Any, IO, List
import logging
from datetime import datetime
import narwhals as nw
from narwhals.typing import IntoFrameT

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

    DEFAULT_BASE_URL = "http://crowdcent.com/api"
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
                      http://crowdcent.com/api.
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
                    error_message = error_data["error"].get("message", e.response.text)
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
        except requests_exceptions.RequestException as e:
            logger.error(f"Request failed: {e} for {method} {url}")
            raise CrowdCentAPIError(f"Request failed: {e}") from e

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

    def get_latest_training_dataset(self) -> Dict[str, Any]:
        """Gets the latest training dataset for this challenge.

        Returns:
            A dictionary representing the latest training dataset.

        Raises:
            NotFoundError: If the challenge or its latest training dataset is not found.
        """
        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/training_data/latest/"
        )
        return response.json()

    def get_training_dataset(self, version: str) -> Dict[str, Any]:
        """Gets details for a specific training dataset version.

        Args:
            version: The version string of the training dataset (e.g., '1.0', '2.1').

        Returns:
            A dictionary representing the specified training dataset.

        Raises:
            NotFoundError: If the challenge or the specified training dataset is not found.
        """
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
            endpoint = (
                f"/challenges/{self.challenge_slug}/training_data/latest/download/"
            )
        else:
            endpoint = (
                f"/challenges/{self.challenge_slug}/training_data/{version}/download/"
            )

        logger.info(
            f"Downloading training data for challenge '{self.challenge_slug}' v{version} to {dest_path}"
        )
        response = self._request("GET", endpoint, stream=True)

        try:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded training data to {dest_path}")
        except IOError as e:
            logger.error(f"Failed to write dataset to {dest_path}: {e}")
            raise CrowdCentAPIError(f"Failed to write dataset file: {e}") from e

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

    def get_current_inference_data(self) -> Dict[str, Any]:
        """Gets the current inference data period for this challenge.

        Returns:
            A dictionary representing the current inference data period.

        Raises:
            NotFoundError: If the challenge has no active inference period.
        """
        response = self._request(
            "GET", f"/challenges/{self.challenge_slug}/inference_data/current/"
        )
        return response.json()

    def get_inference_data(self, release_date: str) -> Dict[str, Any]:
        """Gets details for a specific inference data period by its release date.

        Args:
            release_date: The release date of the inference data in 'YYYY-MM-DD' format.

        Returns:
            A dictionary representing the specified inference data period.

        Raises:
            NotFoundError: If the challenge or the specified inference data is not found.
            ClientError: If the date format is invalid.
        """
        # Validate date format
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

    def download_inference_data(self, release_date: str, dest_path: str):
        """Downloads the inference features file for a specific period.

        Args:
            release_date: The release date of the inference data in 'YYYY-MM-DD' format
                         or 'current' to get the current period's data.
            dest_path: The local file path to save the downloaded features file.

        Raises:
            NotFoundError: If the challenge, inference data, or its file is not found.
            ClientError: If the date format is invalid.
        """
        if release_date == "current":
            endpoint = (
                f"/challenges/{self.challenge_slug}/inference_data/current/download/"
            )
        else:
            # Validate date format
            try:
                datetime.strptime(release_date, "%Y-%m-%d")
            except ValueError:
                raise ClientError(
                    f"Invalid date format: {release_date}. Use 'YYYY-MM-DD' format."
                )

            endpoint = f"/challenges/{self.challenge_slug}/inference_data/{release_date}/download/"

        logger.info(
            f"Downloading inference data for challenge '{self.challenge_slug}' {release_date} to {dest_path}"
        )
        response = self._request("GET", endpoint, stream=True)

        try:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded inference data to {dest_path}")
        except IOError as e:
            logger.error(f"Failed to write inference data to {dest_path}: {e}")
            raise CrowdCentAPIError(f"Failed to write inference data file: {e}") from e

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
        temp: bool = True,
    ) -> Dict[str, Any]:
        """Submits predictions for the current active inference period of this challenge.

        You can provide either a file path to an existing Parquet file or a DataFrame
        that will be temporarily saved as Parquet for submission.

        The data must contain the required prediction columns specified by the challenge
        (e.g., id, pred_10d, pred_30d).

        Args:
            file_path: Optional path to an existing prediction Parquet file.
            df: Optional DataFrame with the prediction columns. If provided,
                it will be temporarily saved as Parquet for submission.
            slot: Submission slot number (1-based).
            temp: Whether to save the DataFrame to a temporary file.

        Returns:
            A dictionary representing the newly created or updated submission.

        Raises:
            ValueError: If neither file_path nor df is provided, or if both are provided.
            FileNotFoundError: If the specified file_path does not exist.
            ClientError: If the submission is invalid (e.g., wrong format,
                         outside submission window, already submitted, etc).

        Examples:
            # Submit from a DataFrame
            client.submit_predictions(df=predictions_df)

            # Submit from a file
            client.submit_predictions(file_path="predictions.parquet")
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
                data_payload = {"slot": str(slot)}
                response = self._request(
                    "POST",
                    f"/challenges/{self.challenge_slug}/submissions/",
                    files=files,
                    data=data_payload,  # Pass slot in data
                )
            logger.info(
                f"Successfully submitted predictions to challenge '{self.challenge_slug}'"
            )
            return response.json()
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

    # --- Meta Model Download ---

    def download_meta_model(self, dest_path: str):
        """Downloads the consolidated meta model file for this challenge.

        The meta model is typically an aggregation (e.g., average) of all valid
        submissions for past inference periods.

        Args:
            dest_path: The local file path to save the downloaded meta model.

        Raises:
            NotFoundError: If the challenge or its meta model file is not found.
            CrowdCentAPIError: For issues during download or file writing.
            PermissionDenied: If the meta model is not public and user lacks permission.
        """
        endpoint = f"/challenges/{self.challenge_slug}/meta_model/download/"
        logger.info(
            f"Downloading consolidated meta model for challenge '{self.challenge_slug}' to {dest_path}"
        )

        # The API endpoint redirects to a signed URL, but requests handles the redirect automatically.
        # We still stream the response from the final URL.
        response = self._request("GET", endpoint, stream=True)

        try:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(
                f"Successfully downloaded consolidated meta model to {dest_path}"
            )
        except IOError as e:
            logger.error(f"Failed to write meta model to {dest_path}: {e}")
            raise CrowdCentAPIError(f"Failed to write meta model file: {e}") from e
