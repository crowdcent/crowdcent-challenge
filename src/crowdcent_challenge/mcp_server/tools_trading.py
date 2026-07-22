"""Trading tools (live Hyperliquid OMS over the CrowdCent API).

Visible in list_tools only when the presenting key has trading capability
(``allow_trading`` + ``oms_access``). Every server-side gate — staff
preview, gross caps, plan-hash consent — binds regardless of what any
client displays. Client exceptions propagate verbatim as tool errors
(CONFIRMATION_REQUIRED, ACCOUNT_BUSY, ...), so act on them literally.

All tools default to network="testnet"; mainnet is an explicit opt-in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .runtime import DEFAULT_CHALLENGE, client_for


def register_trading_tools(mcp) -> None:
    @mcp.tool
    def get_trading_accounts() -> List[Dict[str, Any]]:
        """List the user's Hyperliquid trading accounts (both networks):
        status, can_trade, master_address, agent expiry, builder approval.
        Never any key material."""
        return client_for().get_trading_accounts()

    @mcp.tool
    def get_mandate(
        network: str = "testnet", challenge_slug: str = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Get the live mandate (execution policy + weighted strategy
        sleeves) for this challenge on the given network."""
        return client_for(challenge_slug).get_mandate(network=network)

    @mcp.tool
    def set_mandate(
        mandate: Dict[str, Any],
        network: str = "testnet",
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Create or fully replace the mandate: {"sleeves": [{"config" |
        "config_token", "weight", "label"?}], ...execution knobs}.

        Sleeve configs use the simulation vocabulary — deploy exactly what
        was backtested by passing the winning config_token. ALWAYS show the
        user what you are about to deploy and get their confirmation first.
        """
        return client_for(challenge_slug).set_mandate(mandate, network=network)

    @mcp.tool
    def get_target_book(
        network: str = "testnet", challenge_slug: str = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """The blended target book the mandate currently resolves to:
        target holdings, ranking day, and per-sleeve books."""
        return client_for(challenge_slug).get_target_book(network=network)

    @mcp.tool
    def preview_rebalance(
        network: str = "testnet", challenge_slug: str = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Plan a rebalance WITHOUT executing: trades, turnover, estimated
        fees, plus the plan_hash needed for execution. Always the first
        step — show the user this plan before anything live."""
        return client_for(challenge_slug).preview_rebalance(network=network)

    @mcp.tool
    def execute_rebalance(
        plan_hash: str,
        network: str = "testnet",
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Execute a LIVE rebalance on the user's Hyperliquid account.

        NEVER call this without first calling preview_rebalance, showing
        the user the planned trades/turnover/fees, and receiving their
        explicit confirmation in this conversation. plan_hash comes from
        the preview and expires after 10 minutes. Real orders, real money
        on mainnet.

        On failure surfaces, act literally: ACCOUNT_BUSY means another OMS
        action is running — do not retry immediately. A run reported as
        unknown/held/blocked means stop and tell the user; do not resubmit.
        """
        return client_for(challenge_slug).execute_rebalance(plan_hash, network=network)

    @mcp.tool
    def flatten_positions(
        plan_hash: Optional[str] = None,
        preview: bool = False,
        network: str = "testnet",
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Close EVERY position (full liquidation) — two-step like
        rebalancing: call with preview=true first, show the user the plan,
        then call again with the returned plan_hash after their explicit
        confirmation."""
        return client_for(challenge_slug).flatten(
            plan_hash, preview=preview, network=network
        )

    @mcp.tool
    def pause_trading(
        network: str = "testnet", challenge_slug: str = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Pause the mandate: scheduled trading stops until resumed. Safe
        to call proactively when something looks wrong."""
        return client_for(challenge_slug).pause_trading(network=network)

    @mcp.tool
    def resume_trading(
        network: str = "testnet", challenge_slug: str = DEFAULT_CHALLENGE
    ) -> Dict[str, Any]:
        """Resume the mandate (scheduled trading on). Confirm with the user
        before re-enabling live risk."""
        return client_for(challenge_slug).resume_trading(network=network)

    @mcp.tool
    def list_rebalance_runs(
        limit: int = 10,
        network: str = "testnet",
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> List[Dict[str, Any]]:
        """Recent rebalance runs (the audit trail): kind, status, ranking
        day, result summary, error. The morning-briefing feed."""
        return client_for(challenge_slug).list_rebalance_runs(
            limit=limit, network=network
        )

    @mcp.tool
    def list_open_orders(
        status: Optional[str] = None,
        network: str = "testnet",
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> List[Dict[str, Any]]:
        """The order blotter, newest first, optionally filtered by status
        (resting, filled, twap_running, ...)."""
        return client_for(challenge_slug).list_orders(status=status, network=network)
