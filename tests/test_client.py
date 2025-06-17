import pytest
import requests
from unittest.mock import patch

from crowdcent_challenge.client import (
    ChallengeClient,
    CrowdCentAPIError,
    AuthenticationError,
    NotFoundError,
    ClientError,
    ServerError,
)

BASE_URL = "http://test.crowdcent.com/api"
TEST_SLUG = "test-challenge"
TEST_API_KEY = "test_api_key_123"

# --- Fixtures ---


@pytest.fixture
def client():
    """Provides a ChallengeClient instance initialized with a dummy key."""
    return ChallengeClient(
        challenge_slug=TEST_SLUG, api_key=TEST_API_KEY, base_url=BASE_URL
    )


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Ensure no real API key is accidentally picked up from the environment."""
    monkeypatch.delenv("CROWDCENT_API_KEY", raising=False)


# --- Authentication Tests ---


def test_client_init_with_api_key():
    """Test client initialization with API key provided directly."""
    client = ChallengeClient(
        challenge_slug=TEST_SLUG, api_key=TEST_API_KEY, base_url=BASE_URL
    )
    assert client.challenge_slug == TEST_SLUG
    assert client.api_key == TEST_API_KEY
    assert client.base_url == BASE_URL.rstrip("/")
    assert f"Api-Key {TEST_API_KEY}" in client.session.headers["Authorization"]


def test_client_init_with_env_var(monkeypatch):
    """Test client initialization with API key from environment variable."""
    monkeypatch.setenv("CROWDCENT_API_KEY", "env_api_key")
    client = ChallengeClient(challenge_slug=TEST_SLUG, base_url=BASE_URL)
    assert client.api_key == "env_api_key"
    assert "Api-Key env_api_key" in client.session.headers["Authorization"]


@patch("crowdcent_challenge.client.load_dotenv")
def test_client_init_with_dotenv(mock_load_dotenv, monkeypatch):
    """Test initialization with API key from .env file.

    Since we're testing the client's behavior, we simulate what load_dotenv would do
    rather than actually creating a file on disk.
    """

    # Mock out the environment after load_dotenv would have run
    def set_env_from_dotenv():
        monkeypatch.setenv("CROWDCENT_API_KEY", "dotenv_api_key")
        return True  # Indicate .env was found

    mock_load_dotenv.side_effect = set_env_from_dotenv

    client = ChallengeClient(challenge_slug=TEST_SLUG, base_url=BASE_URL)
    assert client.api_key == "dotenv_api_key"
    assert "Api-Key dotenv_api_key" in client.session.headers["Authorization"]
    mock_load_dotenv.assert_called_once()  # Verify load_dotenv was actually called


@patch("crowdcent_challenge.client.load_dotenv")
def test_client_init_no_key(mock_load_dotenv):
    """Test client initialization fails when no API key is found."""
    # Mock load_dotenv to do nothing (no .env file)
    mock_load_dotenv.return_value = None

    with pytest.raises(AuthenticationError, match="API key not provided"):
        ChallengeClient(challenge_slug=TEST_SLUG, base_url=BASE_URL)


@patch("crowdcent_challenge.client.load_dotenv")
def test_client_init_key_priority(mock_load_dotenv, monkeypatch):
    """Test API key priority: direct > env var > .env."""

    # Simulate .env file being read
    def set_env_from_dotenv():
        monkeypatch.setenv("CROWDCENT_API_KEY_FROM_DOTENV", "dotenv_api_key")
        return True

    mock_load_dotenv.side_effect = set_env_from_dotenv

    # 1. Direct key takes precedence
    monkeypatch.setenv("CROWDCENT_API_KEY", "env_api_key")
    client_direct = ChallengeClient(
        challenge_slug=TEST_SLUG, api_key="direct_key", base_url=BASE_URL
    )
    assert client_direct.api_key == "direct_key"

    # 2. Env var takes precedence over .env
    client_env = ChallengeClient(challenge_slug=TEST_SLUG, base_url=BASE_URL)
    assert client_env.api_key == "env_api_key"  # Not 'dotenv_api_key'

    # 3. .env is used if others aren't present
    monkeypatch.delenv("CROWDCENT_API_KEY")

    # We need a new mock since the API key is loaded during init
    with patch("crowdcent_challenge.client.load_dotenv") as new_mock:

        def set_dotenv_only():
            monkeypatch.setenv("CROWDCENT_API_KEY", "dotenv_api_key")
            return True

        new_mock.side_effect = set_dotenv_only
        client_dotenv = ChallengeClient(challenge_slug=TEST_SLUG, base_url=BASE_URL)
        assert client_dotenv.api_key == "dotenv_api_key"
        new_mock.assert_called_once()


# --- Challenge Methods Tests ---


def test_get_challenge_success(client, requests_mock):
    """Test successful retrieval of challenge details."""
    mock_url = f"{BASE_URL}/challenges/{TEST_SLUG}/"
    mock_data = {"name": "Test Challenge", "slug": TEST_SLUG, "description": "A test"}
    requests_mock.get(mock_url, json=mock_data)

    challenge = client.get_challenge()
    assert challenge == mock_data
    assert (
        requests_mock.last_request.headers["Authorization"] == f"Api-Key {TEST_API_KEY}"
    )


def test_get_challenge_not_found(client, requests_mock):
    """Test handling of 404 error when getting challenge details."""
    mock_url = f"{BASE_URL}/challenges/{TEST_SLUG}/"
    requests_mock.get(
        mock_url, status_code=404, json={"error": {"message": "Not Found"}}
    )

    with pytest.raises(NotFoundError, match="Resource not found"):
        client.get_challenge()


def test_switch_challenge(client):
    """Test switching the client to a new challenge slug."""
    assert client.challenge_slug == TEST_SLUG
    new_slug = "new-test-challenge"
    client.switch_challenge(new_slug)
    assert client.challenge_slug == new_slug


# --- Class Method Tests ---


def test_list_all_challenges_success(requests_mock, monkeypatch):
    """Test successful listing of all challenges via class method."""
    monkeypatch.setenv("CROWDCENT_API_KEY", "class_method_key")
    mock_url = f"{BASE_URL}/challenges/"
    mock_data = [
        {"name": "Challenge 1", "slug": "c1"},
        {"name": "Challenge 2", "slug": "c2"},
    ]
    requests_mock.get(mock_url, json=mock_data)

    challenges = ChallengeClient.list_all_challenges(base_url=BASE_URL)
    assert challenges == mock_data
    assert (
        requests_mock.last_request.headers["Authorization"]
        == "Api-Key class_method_key"
    )


@patch("crowdcent_challenge.client.load_dotenv")
def test_list_all_challenges_no_key(mock_load_dotenv):
    """Test class method fails if no API key is found."""
    # Mock load_dotenv to do nothing (no .env file)
    mock_load_dotenv.return_value = None

    with pytest.raises(AuthenticationError, match="API key not provided"):
        ChallengeClient.list_all_challenges(base_url=BASE_URL)


def test_list_all_challenges_auth_error(requests_mock):
    """Test class method handles authentication error."""
    mock_url = f"{BASE_URL}/challenges/"
    requests_mock.get(
        mock_url, status_code=401, json={"error": {"message": "Invalid key"}}
    )
    with pytest.raises(AuthenticationError, match="Authentication failed"):
        # Provide a dummy key that will be rejected by the mock
        ChallengeClient.list_all_challenges(api_key="rejected_key", base_url=BASE_URL)


# --- Error Handling Tests ---


@pytest.mark.parametrize(
    "status_code, exception_type, error_message",
    [
        (401, AuthenticationError, "Authentication failed"),
        (404, NotFoundError, "Resource not found"),
        (400, ClientError, "Client error"),
        (403, ClientError, "Client error"),  # Another 4xx
        (500, ServerError, "Server error"),
        (503, ServerError, "Server error"),  # Another 5xx
    ],
)
def test_request_error_handling(
    client, requests_mock, status_code, exception_type, error_message
):
    """Test the unified error handling for different HTTP status codes."""
    mock_url = f"{BASE_URL}/challenges/{TEST_SLUG}/"
    requests_mock.get(
        mock_url,
        status_code=status_code,
        json={"error": {"code": "TEST_CODE", "message": "Detailed error"}},
    )

    with pytest.raises(exception_type, match=error_message):
        client.get_challenge()


def test_request_non_json_error(client, requests_mock):
    """Test error handling when the error response is not JSON."""
    mock_url = f"{BASE_URL}/challenges/{TEST_SLUG}/"
    requests_mock.get(mock_url, status_code=400, text="Bad Request Text")

    with pytest.raises(ClientError, match="Bad Request Text"):
        client.get_challenge()


def test_request_connection_error(client, requests_mock):
    """Test handling of network-level errors."""
    mock_url = f"{BASE_URL}/challenges/{TEST_SLUG}/"
    requests_mock.get(mock_url, exc=requests.exceptions.ConnectTimeout)

    with pytest.raises(CrowdCentAPIError, match="Request failed"):
        client.get_challenge()


# --- Add tests for Training Data, Inference Data, Submissions, Downloads, Uploads below ---
# TODO: Add tests for list_training_datasets, get_training_dataset
# TODO: Add tests for download_training_dataset (success, errors, file writing)
# TODO: Add tests for list_inference_data, get_inference_data
# TODO: Add tests for download_inference_data (success, errors, file writing, 'current' date)
# TODO: Add tests for list_submissions, get_submission
# TODO: Add tests for submit_predictions (success, errors, file reading, format validation - mocked)

# --- Additional tests to improve coverage (downloads, date validation) ---


class _DummyStreamResponse:
    """Very small stub of `requests.Response` that only exposes iter_content and headers."""

    def __init__(self, content: bytes = b"dummy-data"):
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def iter_content(self, chunk_size: int = 8192):  # noqa: D401 – simple generator
        yield self._content

    # The client never calls `.json()` on streaming download responses, but we
    # include it to avoid accidental AttributeErrors in other cases.
    def json(self):  # pragma: no cover – not used in current tests
        return {}


def test_download_training_dataset_success(client, tmp_path, monkeypatch):
    """Ensure `download_training_dataset` streams bytes to disk correctly."""
    dest = tmp_path / "train.parquet"

    # Patch the private _request to avoid real HTTP and provide a controllable stream.
    monkeypatch.setattr(
        client, "_request", lambda *a, **k: _DummyStreamResponse(b"bytes")
    )

    client.download_training_dataset(version="1.0", dest_path=str(dest))

    assert dest.read_bytes() == b"bytes"


def test_download_inference_data_success_current(client, tmp_path, monkeypatch):
    """Same as above but for the `current` inference period path."""
    dest = tmp_path / "inf.parquet"
    monkeypatch.setattr(
        client, "_request", lambda *a, **k: _DummyStreamResponse(b"xyz")
    )

    client.download_inference_data(release_date="current", dest_path=str(dest))
    assert dest.read_bytes() == b"xyz"


def test_get_inference_data_invalid_date_raises(client):
    """An invalid date format should raise `ClientError`."""
    with pytest.raises(ClientError):
        client.get_inference_data("not-a-date")


def test_download_inference_data_invalid_date_raises(client, tmp_path):
    """Same validation should happen for download helper."""
    with pytest.raises(ClientError):
        client.download_inference_data(
            release_date="13-2025-01", dest_path=str(tmp_path / "x.parquet")
        )
