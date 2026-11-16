"""Hosted-mode auth and middleware (mcp.crowdcent.com).

Anyone can initialize and list tools without a key, so agents and
directories can see what the server does. Tool calls need a key.

Imports fastmcp at module level — import this only from code paths that
already require the [mcp] extra (build_server, tests).
"""

from __future__ import annotations

import os
import sys

from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.middleware import Middleware
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend
from starlette.authentication import AuthenticationError
from starlette.middleware import Middleware as ASGIMiddleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import JSONResponse

from . import runtime
from .runtime import CLOUD_TOOL_NAMES, TRADING_TOOL_NAMES

KEY_REQUIRED = (
    "This tool needs a CrowdCent API key. Send it as "
    "`Authorization: Bearer <key>`. Get a key at "
    "https://crowdcent.com/profile/settings/."
)


class CrowdCentTokenVerifier(TokenVerifier):
    """Accepts `Authorization: Bearer <CrowdCent API key>`.

    Verifies the key against GET /api/auth/check/ (cheap, cached per-key
    for a minute) and records its capabilities as token claims for the
    per-request trading-tool visibility filter.
    """

    CACHE_TTL_S = 60
    CACHE_MAX_KEYS = 4096

    def __init__(self, api_base: str | None = None):
        super().__init__()
        self._api_base = (
            api_base or os.getenv("CROWDCENT_API_URL") or "https://crowdcent.com/api"
        ).rstrip("/")
        self._cache: dict[str, tuple[float, AccessToken]] = {}

    def _check_key(self, token: str):
        import requests

        try:
            response = requests.get(
                f"{self._api_base}/auth/check/",
                headers={"Authorization": f"Api-Key {token}"},
                timeout=10,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def verify_token(self, token: str):
        import time

        import anyio

        cached = self._cache.get(token)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        data = await anyio.to_thread.run_sync(self._check_key, token)
        if not isinstance(data, dict) or "username" not in data:
            return None
        scopes = []
        if data.get("allow_trading") and data.get("oms_access"):
            scopes.append("trading")
        if data.get("allow_cloud"):
            scopes.append("cloud")
        access = AccessToken(
            token=token,
            client_id=str(data.get("username")),
            scopes=scopes,
            expires_at=None,
            claims=data,
        )
        if len(self._cache) >= self.CACHE_MAX_KEYS:
            self._cache.clear()
        self._cache[token] = (time.monotonic() + self.CACHE_TTL_S, access)
        return access


class _OptionalKeyBackend(BearerAuthBackend):
    """No Authorization header means an anonymous caller. A key that fails
    verification is an error, so a user with a wrong key sees a 401 and
    not a server that silently hides their tools."""

    async def authenticate(self, conn):
        result = await super().authenticate(conn)
        if result is None and conn.headers.get("authorization"):
            raise AuthenticationError("Invalid CrowdCent API key")
        return result


def _reject_invalid_key(conn, exc):
    return JSONResponse(
        {"error": "invalid_token", "error_description": str(exc)},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )


class LenientAcceptMiddleware:
    """Give every POST the Accept header that streamable HTTP requires.

    The spec says clients send `Accept: application/json, text/event-stream`,
    and the transport answers 406 otherwise. Some directory probes (mcptry)
    send it only on `initialize`, so their `tools/list` failed and the
    server was listed as down. Both response types are fine for us."""

    ACCEPT = (b"accept", b"application/json, text/event-stream")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            headers = [(k, v) for k, v in scope["headers"] if k != b"accept"]
            scope = {**scope, "headers": [*headers, self.ACCEPT]}
        await self.app(scope, receive, send)


def optional_auth_middleware(verifier: TokenVerifier) -> list[ASGIMiddleware]:
    """HTTP middleware for the hosted app in place of FastMCP's `auth=`.

    FastMCP's `auth=` puts the whole /mcp endpoint behind a 401, so
    nobody can see the tool list without a key. These layers verify a
    key when one is sent and let anonymous requests through.
    `KeyRequiredMiddleware` then refuses anonymous tool calls."""
    return [
        ASGIMiddleware(LenientAcceptMiddleware),
        ASGIMiddleware(
            AuthenticationMiddleware,
            backend=_OptionalKeyBackend(verifier),
            on_error=_reject_invalid_key,
        ),
        ASGIMiddleware(AuthContextMiddleware),
    ]


class KeyRequiredMiddleware(Middleware):
    """Hosted: anonymous callers can initialize and list tools and prompts,
    but every tool call needs a key. The CrowdCent API also checks the key
    on every call, so this gate gives a clear message, not the security."""

    async def on_call_tool(self, context, call_next):
        if runtime.is_hosted() and not runtime.api_key_for_request():
            raise ToolError(KEY_REQUIRED)
        return await call_next(context)


class TradingAuditMiddleware(Middleware):
    """Client-side audit breadcrumb: every trading tool call goes to stderr
    (tool name, network, plan_hash — never full payloads or keys)."""

    async def on_call_tool(self, context, call_next):
        name = getattr(context.message, "name", None)
        if name in TRADING_TOOL_NAMES:
            arguments = getattr(context.message, "arguments", None) or {}
            print(
                f"[crowdcent-mcp] trading call: {name} "
                f"network={arguments.get('network', 'testnet')} "
                f"plan_hash={arguments.get('plan_hash')}",
                file=sys.stderr,
            )
        return await call_next(context)


class TradingVisibilityMiddleware(Middleware):
    """Hide trading tools from keys that cannot trade (stdio and hosted).

    Visibility is UX, not security — a direct call_tool with a non-enabled
    key still fails server-side at the CrowdCent API."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        if runtime.request_allows_trading():
            return tools
        return [tool for tool in tools if tool.name not in TRADING_TOOL_NAMES]


class CloudVisibilityMiddleware(Middleware):
    """Hide Cloud tools from keys without the allow_cloud flag.

    Same shape and same caveat as trading visibility: the CrowdCent API
    enforces the key flag, the account gate, and owner scoping on every
    call regardless of what list_tools showed."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        if runtime.request_allows_cloud():
            return tools
        return [tool for tool in tools if tool.name not in CLOUD_TOOL_NAMES]
