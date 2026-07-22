"""ChallengeClient, assembled from the per-area API modules.

Import paths are unchanged: `from crowdcent_challenge import ChallengeClient`
and `from crowdcent_challenge.client import ChallengeClient` (plus the
exception names) both keep working.
"""

from ..exceptions import (
    AuthenticationError,
    ClientError,
    CrowdCentAPIError,
    NotFoundError,
    ServerError,
)
from .base import BaseClient
from .data import DataAPI
from .simulator import SimulatorAPI
from .submissions import SubmissionsAPI
from .trading import TradingAPI


class ChallengeClient(DataAPI, SubmissionsAPI, SimulatorAPI, TradingAPI, BaseClient):
    """
    Client for interacting with a specific CrowdCent Challenge.

    Handles authentication and provides methods for accessing challenge data,
    training datasets, inference data, and managing prediction submissions for
    a specific challenge identified by its slug — plus meta-model simulation
    and live trading (staff preview until Trading GA).
    """


__all__ = [
    "ChallengeClient",
    "CrowdCentAPIError",
    "AuthenticationError",
    "NotFoundError",
    "ClientError",
    "ServerError",
]
