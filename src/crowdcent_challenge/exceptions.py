"""Exception hierarchy for the CrowdCent API client.

Every API error carries what the server said, as attributes a script can read
without parsing the message: ``status_code``, ``code`` (the API's error code,
e.g. ``VERSION_CONFLICT``), ``message``, ``fields`` (per-field validation
detail, ``{}`` when none), and ``payload`` (the whole error body).
"""


class CrowdCentAPIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str = "", *, status_code=None, code: str = "", fields=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.fields = fields or {}
        self.payload = payload or {}


class AuthenticationError(CrowdCentAPIError):
    """Exception for authentication issues (401, and 403 with an auth code)."""


class NotFoundError(CrowdCentAPIError):
    """Exception for 404 errors."""


class ClientError(CrowdCentAPIError):
    """Exception for 4xx client errors (excluding 401, 404)."""


class ServerError(CrowdCentAPIError):
    """Exception for 5xx server errors."""
