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
from .cloud import CloudAPI
from .data import DataAPI
from .simulator import SimulatorAPI
from .submissions import SubmissionsAPI
from .trading import TradingAPI


class ChallengeClient(
    DataAPI, SubmissionsAPI, SimulatorAPI, TradingAPI, CloudAPI, BaseClient
):
    """
    Client for interacting with a specific CrowdCent Challenge.

    Handles authentication and provides methods for accessing challenge data,
    training datasets, inference data, and managing prediction submissions for
    a specific challenge identified by its slug — plus meta-model simulation,
    live trading (Challenger tier and above), and CrowdCent Cloud hosted
    notebooks (in pilot for members with a submission on the board).
    """


__all__ = [
    "ChallengeClient",
    "CrowdCentAPIError",
    "AuthenticationError",
    "NotFoundError",
    "ClientError",
    "ServerError",
]
