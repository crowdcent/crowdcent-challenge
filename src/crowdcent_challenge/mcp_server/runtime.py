"""Per-request plumbing for the MCP server.

Two modes for one codebase:

- **stdio** (default): one user per process; the API key comes from the
  `CROWDCENT_API_KEY` environment variable.
- **hosted** (`CROWDCENT_MCP_MODE=hosted`): multi-tenant behind bearer auth;
  the API key is the request's bearer token.

Trading tools are always registered; visibility in ``list_tools`` follows the
presenting key's capabilities (``allow_trading`` + ``oms_access`` from
``/auth/check/``). The CrowdCent API enforces every gate server-side.
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

_stdio_trading_cache: tuple[str, float, bool] | None = None


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


def _stdio_trading_allowed() -> bool:
    global _stdio_trading_cache

    api_key = os.getenv("CROWDCENT_API_KEY")
    if not api_key:
        return False
    now = time.monotonic()
    if (
        _stdio_trading_cache is not None
        and _stdio_trading_cache[0] == api_key
        and _stdio_trading_cache[1] > now
    ):
        return _stdio_trading_cache[2]
    try:
        claims = ChallengeClient(
            DEFAULT_CHALLENGE,
            api_key=api_key,
            base_url=api_base_url(),
        ).check_auth()
        allowed = _claims_allow_trading(claims)
    except Exception:
        allowed = False
    _stdio_trading_cache = (api_key, now + _CAP_CHECK_TTL_S, allowed)
    return allowed


def request_allows_trading() -> bool:
    """Whether trading tools should appear for the presenting key."""
    if is_hosted():
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
            claims = (token.claims or {}) if token is not None else {}
            return _claims_allow_trading(claims)
        except Exception:
            return False
    return _stdio_trading_allowed()
