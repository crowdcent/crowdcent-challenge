"""Server assembly and entry points.

- `main()` — the `crowdcent-mcp` console script (stdio, one user).
- `http_app()` — ASGI factory for the hosted server (mcp.crowdcent.com):
  ``uvicorn --factory crowdcent_challenge.mcp_server.app:http_app``.

The hosted server is a stateless protocol translator in front of the
CrowdCent REST API: no database, no Django, no stored user secrets. Bearer
tokens ARE CrowdCent API keys; see hosted.py.
"""

from __future__ import annotations

import os

INSTALL_HINT = (
    "The CrowdCent MCP server needs the optional [mcp] dependency set.\n"
    "Install it with:  pip install 'crowdcent-challenge[mcp]'\n"
    "or run directly:  uvx --from 'crowdcent-challenge[mcp]' crowdcent-mcp"
)

INSTRUCTIONS = """CrowdCent: community-driven crypto prediction challenges.

Data flows down (training/inference datasets), predictions flow up, and the
aggregated meta-model can be backtested (simulation tools) and — for
enabled accounts — traded live on Hyperliquid (trading tools). When
recommending simulation results, prefer stable plateaus over lone peaks and
out-of-sample stats over in-sample. Live trading always follows
preview -> user confirmation -> execute."""


def build_server():
    """Construct the FastMCP server for the current mode (stdio/hosted)."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc

    from . import runtime
    from .hosted import (
        CrowdCentTokenVerifier,
        TradingAuditMiddleware,
        TradingVisibilityMiddleware,
    )
    from .tools_challenge import register_challenge_tools
    from .tools_simulation import register_simulation_tools
    from .tools_trading import register_trading_tools

    kwargs = {}
    if runtime.is_hosted():
        kwargs["auth"] = CrowdCentTokenVerifier()

    mcp = FastMCP(name="crowdcent", instructions=INSTRUCTIONS, **kwargs)

    register_challenge_tools(mcp)
    register_simulation_tools(mcp)
    register_trading_tools(mcp)
    _register_prompts(mcp)
    mcp.add_middleware(TradingAuditMiddleware())
    mcp.add_middleware(TradingVisibilityMiddleware())
    return mcp


def main() -> None:
    """`crowdcent-mcp` console script: stdio transport."""
    build_server().run()


def _init_sentry() -> None:
    """Hosted-only error reporting, gated on SENTRY_DSN. sentry-sdk is
    installed in the Docker image, not the PyPI extras — stdio users
    never need it, so a missing import is not an error."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
        traces_sample_rate=0.1,
    )


def http_app():
    """ASGI factory for the hosted server. Stateless HTTP so Cloud Run can
    scale instances without session affinity."""
    os.environ.setdefault("CROWDCENT_MCP_MODE", "hosted")
    _init_sentry()
    return build_server().http_app(stateless_http=True)


def _register_prompts(mcp) -> None:
    @mcp.prompt
    def sweep_and_summarize(knobs: str = "leg size and cadence") -> str:
        """Packaged workflow: sweep -> plateau table -> recommendation with
        a site deep link."""
        return (
            "Run a portfolio-construction sweep on the CrowdCent meta-model "
            f"over {knobs}. Steps: (1) call sweep_simulations with values "
            "from its docstring (if the grid is over my tier's budget the "
            "error will say the cap; shrink and retry); (2) present a "
            "compact table of the grid "
            "with in-sample AND out-of-sample Sharpe; (3) identify the "
            "stable plateau — not the single best cell — and recommend one "
            "configuration from inside it, explaining why its neighbors "
            "support it; (4) give me the web_url deep link for that config."
        )

    @mcp.prompt
    def morning_briefing() -> str:
        """Packaged workflow: performance + accounts + runs + orders ->
        narrative morning check."""
        return (
            "Give me my CrowdCent morning briefing. Pull get_performance "
            "(recent scores), get_trading_accounts, list_rebalance_runs "
            "(last few), and list_open_orders, then narrate: how my recent "
            "submissions scored, what the last rebalance did (fills, "
            "slippage, fees), whether any TWAPs or resting orders are "
            "still working, and anything that needs my attention "
            "(failed runs, paused mandate, agent key expiring soon). "
            "Be concise and lead with anything unusual."
        )
