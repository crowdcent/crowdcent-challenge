"""Per-request plumbing for the MCP server.

Two modes for one codebase:

- **stdio** (default): one user per process; the API key comes from the
  `CROWDCENT_API_KEY` environment variable.
- **hosted** (`CROWDCENT_MCP_MODE=hosted`): multi-tenant behind bearer auth;
  the API key is the request's bearer token.

Trading and Cloud tools are always registered; visibility in ``list_tools``
follows the presenting key's capabilities (``allow_trading`` + ``oms_access``
and ``allow_cloud`` from ``/auth/check/``). The CrowdCent API enforces every
gate server-side.
"""

from __future__ import annotations

import os
import time

from crowdcent_challenge import ChallengeClient
from crowdcent_challenge.exceptions import AuthenticationError

DEFAULT_CHALLENGE = "hyperliquid-ranking"
_CAP_CHECK_TTL_S = 60

TRADING_TOOL_NAMES = frozenset(
    {
        "get_trading_accounts",
        "get_mandate",
        "set_mandate",
        "get_target_book",
        "preview_rebalance",
        "execute_rebalance",
        "flatten_positions",
        "pause_trading",
        "resume_trading",
        "list_rebalance_runs",
        "list_open_orders",
    }
)

CLOUD_TOOL_NAMES = frozenset(
    {
        "get_cloud_billing",
        "update_cloud_billing",
        "list_cloud_recipes",
        "list_cloud_projects",
        "get_cloud_project",
        "get_cloud_project_files",
        "download_cloud_project_file",
        "upload_cloud_project_files",
        "create_cloud_project",
        "update_cloud_project",
        "archive_cloud_project",
        "run_cloud_project",
        "get_cloud_run",
        "schedule_cloud_project",
        "pause_cloud_project_schedule",
    }
)

_stdio_claims_cache: tuple[str, float, dict] | None = None


def is_hosted() -> bool:
    return os.getenv("CROWDCENT_MCP_MODE") == "hosted"


def api_base_url() -> str | None:
    """Optional API base override (tests, staging); None = client default."""
    return os.getenv("CROWDCENT_API_URL") or None


def api_key_for_request() -> str | None:
    """The CrowdCent API key for THIS tool call.

    Hosted: the request's bearer token (verified at connect time by the
    TokenVerifier) simply passes through to the REST API, which
    authenticates every call itself. Stdio: the process env var.
    """
    if is_hosted():
        from fastmcp.server.dependencies import get_access_token

        token = get_access_token()
        return token.token if token is not None else None
    return os.getenv("CROWDCENT_API_KEY")


def client_for(challenge_slug: str = DEFAULT_CHALLENGE) -> ChallengeClient:
    """A ChallengeClient bound to this request's key and the given challenge.

    Resolved per tool call — never a module-level global — so one hosted
    process serves many users and stdio behaves identically.
    """
    api_key = api_key_for_request()
    if is_hosted() and not api_key:
        raise AuthenticationError("Bearer token required")
    return ChallengeClient(
        challenge_slug,
        api_key=api_key,
        base_url=api_base_url(),
    )


def _claims_allow_trading(claims: dict) -> bool:
    return bool(claims.get("allow_trading")) and bool(claims.get("oms_access"))


def _claims_allow_cloud(claims: dict) -> bool:
    return bool(claims.get("allow_cloud"))


def _stdio_claims() -> dict:
    """The stdio key's /auth/check/ capabilities, cached briefly."""
    global _stdio_claims_cache

    api_key = os.getenv("CROWDCENT_API_KEY")
    if not api_key:
        return {}
    now = time.monotonic()
    if (
        _stdio_claims_cache is not None
        and _stdio_claims_cache[0] == api_key
        and _stdio_claims_cache[1] > now
    ):
        return _stdio_claims_cache[2]
    try:
        claims = ChallengeClient(
            DEFAULT_CHALLENGE,
            api_key=api_key,
            base_url=api_base_url(),
        ).check_auth()
    except Exception:
        claims = {}
    _stdio_claims_cache = (api_key, now + _CAP_CHECK_TTL_S, claims)
    return claims


def _request_claims() -> dict:
    """This request's key capabilities: token claims (hosted) or the
    cached /auth/check/ answer (stdio)."""
    if is_hosted():
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
            return (token.claims or {}) if token is not None else {}
        except Exception:
            return {}
    return _stdio_claims()


def request_allows_trading() -> bool:
    """Whether trading tools should appear for the presenting key."""
    return _claims_allow_trading(_request_claims())


def request_allows_cloud() -> bool:
    """Whether Cloud tools should appear for the presenting key."""
    return _claims_allow_cloud(_request_claims())
