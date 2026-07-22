"""Trading (staff preview until Trading GA): mandate, target book,
preview/execute consent flow, pause/resume, audit feeds."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional  # noqa: F401

logger = logging.getLogger(__name__)


class TradingAPI:
    # --- Trading (staff preview until Trading GA) ---
    #
    # Every method takes network="testnet" | "mainnet" and DEFAULTS TO
    # TESTNET — mainnet is always an explicit opt-in. The server has no
    # default at all. Live execution is two-step: preview_rebalance() returns
    # a plan plus a plan_hash; execute_rebalance(plan_hash) consumes it
    # within 10 minutes. The hash is consent evidence, not an execution
    # contract — the server re-plans fresh and its caps still bind.

    def get_trading_accounts(self) -> List[Dict[str, Any]]:
        """Lists your Hyperliquid trading accounts (both networks).

        Returns only safe fields (`status`, `can_trade`, `master_address`,
        `agent_expires_at`, `builder_approved`, `network`) — key material
        never traverses the API.
        """
        response = self._request("GET", "/trading/accounts/")
        return response.json()["accounts"]

    def get_mandate(self, network: str = "testnet") -> Dict[str, Any]:
        """Gets the mandate (execution policy + weighted sleeves) for this
        challenge on the given network.

        Raises:
            NotFoundError: `NO_MANDATE` if none exists yet — create one with
                :py:meth:`set_mandate`.
        """
        response = self._request(
            "GET",
            f"/challenges/{self.challenge_slug}/trading/mandate/",
            params={"network": network},
        )
        return response.json()

    def set_mandate(
        self, mandate: Dict[str, Any], network: str = "testnet"
    ) -> Dict[str, Any]:
        """Creates or fully replaces the mandate for this challenge/network.

        Args:
            mandate: ``{"sleeves": [{"config" | "config_token", "weight",
                "label"?}, ...], ...execution knobs...}``. Sleeve configs use
                the simulation vocabulary and validate through the same
                parser and tier locks as the site. Optional execution knobs
                (`order_type`, `target_leverage`, `schedule_enabled`,
                `schedule_at_time`, `twap_minutes`, `stop_loss_pct`, ...)
                clamp to the web's bounds.
            network: "testnet" (default) or "mainnet".

        Returns:
            The saved mandate, same shape as :py:meth:`get_mandate`.
        """
        response = self._request(
            "PUT",
            f"/challenges/{self.challenge_slug}/trading/mandate/",
            json_data={**mandate, "network": network},
        )
        return response.json()

    def get_target_book(self, network: str = "testnet") -> Dict[str, Any]:
        """Gets the blended target book the mandate's sleeves currently
        resolve to: target holdings, `as_of` ranking day, and per-sleeve
        books."""
        response = self._request(
            "GET",
            f"/challenges/{self.challenge_slug}/trading/book/",
            params={"network": network},
        )
        return response.json()

    def preview_rebalance(self, network: str = "testnet") -> Dict[str, Any]:
        """Plans a rebalance WITHOUT executing (dry run, persisted for audit).

        Returns the full plan (`trades`, `skipped_trades`, `gross`,
        `turnover`, `est_fees`) plus `run_id` and **`plan_hash`** — show the
        plan to the user, get their explicit confirmation, then pass the
        hash to :py:meth:`execute_rebalance` within 10 minutes.
        """
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/trading/rebalance/preview/",
            json_data={"network": network},
        )
        return response.json()

    def execute_rebalance(
        self, plan_hash: str, network: str = "testnet"
    ) -> Dict[str, Any]:
        """Executes a LIVE rebalance on your Hyperliquid account.

        Real orders; real money on mainnet. Requires a fresh (<10 min)
        preview's `plan_hash` as consent evidence, else the server answers
        409 `CONFIRMATION_REQUIRED`. Execution re-plans fresh, so fills may
        differ slightly from the preview; the gross cap, minimum notional,
        and stale-book guards always bind server-side.

        Raises:
            ClientError: `CONFIRMATION_REQUIRED` (409) without a fresh
                matching hash; `ACCOUNT_BUSY` (409) if another OMS action
                holds the account lock — do not retry immediately.
        """
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/trading/rebalance/",
            json_data={"network": network, "plan_hash": plan_hash},
        )
        return response.json()

    def flatten(
        self,
        plan_hash: Optional[str] = None,
        *,
        preview: bool = False,
        network: str = "testnet",
    ) -> Dict[str, Any]:
        """Closes every position (full liquidation) — two-step like
        rebalancing.

        Call with ``preview=True`` first to get the flatten plan and its
        `plan_hash`; then call again with that hash to execute. Flatten
        hashes are kind-scoped: a rebalance preview's hash never authorizes
        a flatten.
        """
        if preview:
            payload: Dict[str, Any] = {"network": network, "preview": True}
        else:
            payload = {"network": network, "plan_hash": plan_hash}
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/trading/flatten/",
            json_data=payload,
        )
        return response.json()

    def pause_trading(self, network: str = "testnet") -> Dict[str, Any]:
        """Pauses the mandate (scheduled trading off). Works with ANY valid
        key — killing risk is never blocked by the trading scope."""
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/trading/pause/",
            json_data={"network": network},
        )
        return response.json()

    def resume_trading(self, network: str = "testnet") -> Dict[str, Any]:
        """Resumes the mandate (scheduled trading on). Requires a
        trade-enabled key — re-enabling risk is a trading action."""
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/trading/resume/",
            json_data={"network": network},
        )
        return response.json()

    def list_rebalance_runs(
        self, limit: int = 10, network: str = "testnet"
    ) -> List[Dict[str, Any]]:
        """Lists recent rebalance runs (the audit trail): kind, status,
        release_date, result summary, error. The morning-briefing feed."""
        response = self._request(
            "GET",
            f"/challenges/{self.challenge_slug}/trading/runs/",
            params={"network": network, "limit": limit},
        )
        return response.json()["runs"]

    def list_orders(
        self, status: Optional[str] = None, network: str = "testnet"
    ) -> List[Dict[str, Any]]:
        """Lists blotter orders, newest first, optionally filtered by
        status (resting, filled, twap_running, ...)."""
        params: Dict[str, Any] = {"network": network}
        if status:
            params["status"] = status
        response = self._request(
            "GET",
            f"/challenges/{self.challenge_slug}/trading/orders/",
            params=params,
        )
        return response.json()["orders"]
