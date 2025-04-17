import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any, IO, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class CrowdcentAPIError(Exception):
    """Base exception for API errors."""

    pass


class AuthenticationError(CrowdcentAPIError):
    """Exception for authentication issues."""

    pass


class NotFoundError(CrowdcentAPIError):
    """Exception for 404 errors."""

    pass


class ClientError(CrowdcentAPIError):
    """Exception for 4xx client errors (excluding 401, 404)."""

    pass


class ServerError(CrowdcentAPIError):
    """Exception for 5xx server errors."""

    pass


class ChallengeClient:
    """
    Client for interacting with the Crowdcent Challenge API.

    Handles authentication and provides methods for accessing challenges,
    training datasets, inference data, and managing prediction submissions.
    """

    DEFAULT_BASE_URL = "http://crowdcent.com/api"
    API_KEY_ENV_VAR = "CROWDCENT_API_KEY"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initializes the ChallengeClient.

        Args:
            api_key: Your Crowdcent API key. If not provided, it will attempt
                     to load from the CROWDCENT_API_KEY environment variable
                     or a .env file.
            base_url: The base URL of the Crowdcent API. Defaults to
                      http://crowdcent.com/api.
        """
        load_dotenv()  # Load .env file if present
        self.api_key = api_key or os.getenv(self.API_KEY_ENV_VAR)
        if not self.api_key:
            raise AuthenticationError(
                f"API key not provided and not found in environment variable "
                f"'{self.API_KEY_ENV_VAR}' or .env file."
            )

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Api-Key {self.api_key}"})
        logger.info(f"ChallengeClient initialized for URL: {self.base_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        files: Optional[Dict[str, IO]] = None,
        stream: bool = False,
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

        Returns:
            The requests.Response object.

        Raises:
            AuthenticationError: If the API key is invalid (401).
            NotFoundError: If the resource is not found (404).
            ClientError: For other 4xx errors.
            ServerError: For 5xx errors.
            CrowdcentAPIError: For other request exceptions.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(
            f"Request: {method} {url} Params: {params} JSON: {json_data is not None} Files: {files is not None}"
        )

        try:
            response = self.session.request(
                method, url, params=params, json=json_data, files=files, stream=stream
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.debug(f"Response: {response.status_code}")
            return response
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            # Try to parse standardized error format: {"error": {"code": "ERROR_CODE", "message": "Description"}}
            try:
                error_data = e.response.json()
                if 'error' in error_data and isinstance(error_data['error'], dict):
                    error_code = error_data['error'].get('code', 'UNKNOWN_ERROR')
                    error_message = error_data['error'].get('message', e.response.text)
                else:
                    error_code = "API_ERROR"
                    error_message = e.response.text
            except requests.exceptions.JSONDecodeError:
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
                raise CrowdcentAPIError(
                    f"HTTP error ({status_code}): {error_message} [{error_code}]"
                ) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e} for {method} {url}")
            raise CrowdcentAPIError(f"Request failed: {e}") from e

    # --- Challenge Methods ---

    def list_challenges(self) -> List[Dict[str, Any]]:
        """Lists all active challenges.

        Returns:
            A list of dictionaries, each representing an active challenge.
        """
        response = self._request("GET", "/challenges/")
        return response.json()

    def get_challenge(self, challenge_slug: str) -> Dict[str, Any]:
        """Gets details for a specific challenge by its slug.

        Args:
            challenge_slug: The slug of the challenge to retrieve.

        Returns:
            A dictionary representing the specified challenge.

        Raises:
            NotFoundError: If the challenge with the given slug is not found.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/")
        return response.json()

    # --- Training Data Methods ---

    def list_training_datasets(self, challenge_slug: str) -> List[Dict[str, Any]]:
        """Lists all training dataset versions for a specific challenge.

        Args:
            challenge_slug: The slug of the challenge.

        Returns:
            A list of dictionaries, each representing a training dataset version.

        Raises:
            NotFoundError: If the challenge with the given slug is not found.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/training_data/")
        return response.json()

    def get_latest_training_dataset(self, challenge_slug: str) -> Dict[str, Any]:
        """Gets the latest training dataset for a specific challenge.

        Args:
            challenge_slug: The slug of the challenge.

        Returns:
            A dictionary representing the latest training dataset.

        Raises:
            NotFoundError: If the challenge or its latest training dataset is not found.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/training_data/latest/")
        return response.json()

    def get_training_dataset(self, challenge_slug: str, version: str) -> Dict[str, Any]:
        """Gets details for a specific training dataset version.

        Args:
            challenge_slug: The slug of the challenge.
            version: The version string of the training dataset (e.g., '1.0', '2.1').

        Returns:
            A dictionary representing the specified training dataset.

        Raises:
            NotFoundError: If the challenge or the specified training dataset is not found.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/training_data/{version}/")
        return response.json()

    def download_training_dataset(self, challenge_slug: str, version: str, dest_path: str):
        """Downloads the training data file for a specific dataset version.

        Args:
            challenge_slug: The slug of the challenge.
            version: The version string of the training dataset (e.g., '1.0', '2.1')
                    or 'latest' to get the latest version.
            dest_path: The local file path to save the downloaded dataset.

        Raises:
            NotFoundError: If the challenge, dataset, or its file is not found.
        """
        if version == 'latest':
            endpoint = f"/challenges/{challenge_slug}/training_data/latest/download/"
        else:
            endpoint = f"/challenges/{challenge_slug}/training_data/{version}/download/"
            
        logger.info(f"Downloading training data {challenge_slug} v{version} to {dest_path}")
        response = self._request("GET", endpoint, stream=True)
        
        try:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded training data to {dest_path}")
        except IOError as e:
            logger.error(f"Failed to write dataset to {dest_path}: {e}")
            raise CrowdcentAPIError(f"Failed to write dataset file: {e}") from e

    # --- Inference Data Methods ---

    def list_inference_data(self, challenge_slug: str) -> List[Dict[str, Any]]:
        """Lists all inference data periods for a specific challenge.

        Args:
            challenge_slug: The slug of the challenge.

        Returns:
            A list of dictionaries, each representing an inference data period.

        Raises:
            NotFoundError: If the challenge with the given slug is not found.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/inference_data/")
        return response.json()

    def get_current_inference_data(self, challenge_slug: str) -> Dict[str, Any]:
        """Gets the current inference data period for a specific challenge.

        Args:
            challenge_slug: The slug of the challenge.

        Returns:
            A dictionary representing the current inference data period.

        Raises:
            NotFoundError: If the challenge has no active inference period.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/inference_data/current/")
        return response.json()

    def get_inference_data(self, challenge_slug: str, release_date: str) -> Dict[str, Any]:
        """Gets details for a specific inference data period by its release date.

        Args:
            challenge_slug: The slug of the challenge.
            release_date: The release date of the inference data in 'YYYY-MM-DD' format.

        Returns:
            A dictionary representing the specified inference data period.

        Raises:
            NotFoundError: If the challenge or the specified inference data is not found.
            ClientError: If the date format is invalid.
        """
        # Validate date format
        try:
            datetime.strptime(release_date, '%Y-%m-%d')
        except ValueError:
            raise ClientError(f"Invalid date format: {release_date}. Use 'YYYY-MM-DD' format.")
            
        response = self._request("GET", f"/challenges/{challenge_slug}/inference_data/{release_date}/")
        return response.json()

    def download_inference_data(self, challenge_slug: str, release_date: str, dest_path: str):
        """Downloads the inference features file for a specific period.

        Args:
            challenge_slug: The slug of the challenge.
            release_date: The release date of the inference data in 'YYYY-MM-DD' format
                         or 'current' to get the current period's data.
            dest_path: The local file path to save the downloaded features file.

        Raises:
            NotFoundError: If the challenge, inference data, or its file is not found.
            ClientError: If the date format is invalid.
        """
        if release_date == 'current':
            endpoint = f"/challenges/{challenge_slug}/inference_data/current/download/"
        else:
            # Validate date format
            try:
                datetime.strptime(release_date, '%Y-%m-%d')
            except ValueError:
                raise ClientError(f"Invalid date format: {release_date}. Use 'YYYY-MM-DD' format.")
                
            endpoint = f"/challenges/{challenge_slug}/inference_data/{release_date}/download/"
            
        logger.info(f"Downloading inference data {challenge_slug} {release_date} to {dest_path}")
        response = self._request("GET", endpoint, stream=True)
        
        try:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded inference data to {dest_path}")
        except IOError as e:
            logger.error(f"Failed to write inference data to {dest_path}: {e}")
            raise CrowdcentAPIError(f"Failed to write inference data file: {e}") from e

    # --- Submission Methods ---

    def list_submissions(self, challenge_slug: str, period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists the authenticated user's submissions for a specific challenge.

        Args:
            challenge_slug: The slug of the challenge.
            period: Optional filter for submissions by period:
                  - 'current': Only show submissions for the current active period
                  - 'YYYY-MM-DD': Only show submissions for a specific inference period date

        Returns:
            A list of dictionaries, each representing a submission.
        """
        params = {}
        if period:
            params['period'] = period
            
        response = self._request(
            "GET", 
            f"/challenges/{challenge_slug}/submissions/", 
            params=params
        )
        return response.json()

    def get_submission(self, challenge_slug: str, submission_id: int) -> Dict[str, Any]:
        """Gets details for a specific submission by its ID.

        Args:
            challenge_slug: The slug of the challenge.
            submission_id: The ID of the submission to retrieve.

        Returns:
            A dictionary representing the specified submission.

        Raises:
            NotFoundError: If the submission with the given ID is not found
                           or doesn't belong to the user.
        """
        response = self._request("GET", f"/challenges/{challenge_slug}/submissions/{submission_id}/")
        return response.json()

    def submit_predictions(self, challenge_slug: str, file_path: str) -> Dict[str, Any]:
        """Submits a prediction file for the current active inference period of a challenge.

        The file must be a Parquet file with the required prediction columns:
        id, pred_1M, pred_3M, pred_6M, pred_9M, pred_12M

        Args:
            challenge_slug: The slug of the challenge to submit predictions for.
            file_path: The path to the prediction Parquet file.

        Returns:
            A dictionary representing the newly created submission.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
            ClientError: If the submission is invalid (e.g., wrong format,
                         outside submission window, already submitted, etc).
        """
        logger.info(f"Submitting predictions from {file_path} to challenge {challenge_slug}")
        try:
            with open(file_path, "rb") as f:
                files = {
                    "prediction_file": (
                        os.path.basename(file_path), 
                        f, 
                        "application/octet-stream"
                    )
                }
                response = self._request(
                    "POST", 
                    f"/challenges/{challenge_slug}/submissions/", 
                    files=files
                )
            logger.info(f"Successfully submitted predictions to challenge {challenge_slug}")
            return response.json()
        except FileNotFoundError as e:
            logger.error(f"Prediction file not found at {file_path}")
            raise FileNotFoundError(f"Prediction file not found at {file_path}") from e
        except IOError as e:
            logger.error(f"Failed to read prediction file {file_path}: {e}")
            raise CrowdcentAPIError(f"Failed to read prediction file: {e}") from e
