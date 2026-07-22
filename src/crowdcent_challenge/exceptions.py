"""Exception hierarchy for the CrowdCent API client."""


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
