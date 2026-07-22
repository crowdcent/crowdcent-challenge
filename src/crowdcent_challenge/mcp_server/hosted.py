"""Hosted-mode auth and middleware (mcp.crowdcent.com).

Imports fastmcp at module level — import this only from code paths that
already require the [mcp] extra (build_server, tests).
"""

from __future__ import annotations

import os
import sys

from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.middleware import Middleware

from . import runtime
from .runtime import TRADING_TOOL_NAMES


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
        access = AccessToken(
            token=token,
            client_id=str(data.get("username")),
            scopes=(
                ["trading"]
                if data.get("allow_trading") and data.get("oms_access")
                else []
            ),
            expires_at=None,
            claims=data,
        )
        if len(self._cache) >= self.CACHE_MAX_KEYS:
            self._cache.clear()
        self._cache[token] = (time.monotonic() + self.CACHE_TTL_S, access)
        return access


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
