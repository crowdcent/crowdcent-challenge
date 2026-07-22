"""Simulation: backtest, sweep, and blend portfolio constructions on
the live meta-model."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional  # noqa: F401

logger = logging.getLogger(__name__)


class SimulatorAPI:
    # --- Simulation ---

    def get_simulator_capabilities(self) -> Dict[str, Any]:
        """Gets the simulator knob vocabulary for YOUR tier.

        Call this before building simulation configs: it lists the allowed
        values per knob for the presenting key's points tier, the sweep and
        blend budgets, the available data range, and which features are
        sealed behind higher tiers.

        Returns:
            A dictionary with `tier`, `data`, `config` (allowed values per
            knob), `sealed_features`, `sweep` (budgets + sweepable knobs),
            `blend`, `benchmark_trials`, and `include_options`.
        """
        response = self._request("GET", f"/challenges/{self.challenge_slug}/simulator/")
        return response.json()

    def run_simulation(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_token: Optional[str] = None,
        include: Optional[List[str]] = None,
        benchmark_trials: int = 0,
    ) -> Dict[str, Any]:
        """Backtests one portfolio configuration on the live meta-model.

        Runs the exact engine behind the site's Simulation tab: the
        simulator trades the meta-model's published rankings as a long/short
        portfolio with your chosen construction knobs (cohort sizes,
        rebalance cadence, weighting scheme, fees, funding, ...).

        Knobs above your tier are silently clamped to their accessible
        values, identical to the web UI. Check the echoed `config` and the
        `locked` list in the response to see what was clamped.

        Args:
            config: SimulationConfig field names with JSON scalars, e.g.
                ``{"n_long": 10, "n_short": 10, "rebalance_days": "10t",
                "weighting": "inv_vol", "include_funding": True}``. Omitted
                knobs use the site's defaults.
            config_token: Alternatively, a compact config token from a
                previous response or a site URL — reproduces that exact
                config. Mutually exclusive with `config`.
            include: Optional extras: any of `"curve"` (daily series),
                `"holdings"` (current book), `"monthly"` (returns grid),
                `"contributions"` (per-asset P&L attribution). Defaults to
                none, keeping responses compact.
            benchmark_trials: 0, 25, or 100 — score the signal against that
                many random-ranking portfolios with identical construction.

        Returns:
            A dictionary with the clamped `config` echo, `locked`,
            `config_token`, `web_url` (the site pre-loaded with this exact
            config), `as_of`, `n_days`, `stats`, `is_stats`, `oos_stats`,
            plus any `include` extras and `benchmark` results.

        Example:
            ```python
            result = client.run_simulation(
                config={"n_long": 10, "n_short": 10, "weighting": "inv_vol"}
            )
            result["stats"]["sharpe"]  # 1.42
            ```
        """
        payload: Dict[str, Any] = {}
        if config is not None:
            payload["config"] = config
        if config_token is not None:
            payload["config_token"] = config_token
        if include:
            payload["include"] = include
        if benchmark_trials:
            payload["benchmark_trials"] = benchmark_trials
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/simulator/run/",
            json_data=payload,
        )
        return response.json()

    def run_sweep(
        self,
        config: Dict[str, Any],
        sweep: Dict[str, List[Any]],
        *,
        on_chunk: Optional[Callable[[List[Dict[str, Any]], int], None]] = None,
    ) -> Dict[str, Any]:
        """Grid-searches portfolio configurations on the meta-model.

        The server runs one cost-bounded chunk per request; this method
        loops `offset` -> `next_offset` transparently until the whole grid
        has run, so you always get the complete result set back.

        How to read a sweep: **plateaus, not peaks.** A lone bright cell is
        luck; a bright region is structure. Prefer configurations whose
        neighbors also perform, and weight out-of-sample stats (`oos_stats`)
        over in-sample when picking a candidate.

        Args:
            config: The base configuration (same vocabulary as
                :py:meth:`run_simulation`); swept knobs override it.
            sweep: Mapping of sweepable knob -> list of values, e.g.
                ``{"n_long": [5, 10, 20], "rebalance_days": ["5t", "10t"]}``.
                See :py:meth:`get_simulator_capabilities` for your tier's
                sweepable knobs, allowed values, and grid budget (96 configs
                at Contender tier, 24 below).
            on_chunk: Optional callable ``on_chunk(results_so_far, total)``
                invoked after each server chunk — useful for progress bars.

        Returns:
            A dictionary with `total` and `results`: one entry per grid cell
            (in deterministic grid order), each holding `config_token`,
            `params` (the swept values), and `stats`/`is_stats`/`oos_stats`
            or an `error`.

        Raises:
            ClientError: `SWEEP_TOO_LARGE` if the grid exceeds your tier's
                budget; `INVALID_SWEEP_KNOB`/`TIER_REQUIRED` for bad knobs.
        """
        results: List[Dict[str, Any]] = []
        offset = 0
        total = None
        while True:
            response = self._request(
                "POST",
                f"/challenges/{self.challenge_slug}/simulator/sweep/",
                json_data={"config": config, "sweep": sweep, "offset": offset},
            )
            body = response.json()
            total = body["total"]
            results.extend(body["results"])
            if on_chunk is not None:
                on_chunk(results, total)
            if body.get("next_offset") is None:
                break
            offset = body["next_offset"]
        return {"total": total, "results": results}

    def run_blend(self, sleeves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Blends weighted sleeves into one ensemble book and evaluates it.

        Each sleeve runs once and the weighted blend is formed on
        date-aligned daily returns. Fail-closed: any sleeve error fails the
        whole blend. Sleeve count is capped by tier (5 at Contender, 3
        below).

        Args:
            sleeves: A list of ``{"config": {...} | "config_token": "...",
                "weight": float, "label": str?}`` dictionaries.

        Returns:
            A dictionary with blend `stats`/`is_stats`/`oos_stats`, the
            sleeve `correlation` matrix, and per-sleeve stats (computed on
            the aligned window so they are directly comparable).
        """
        response = self._request(
            "POST",
            f"/challenges/{self.challenge_slug}/simulator/blend/",
            json_data={"sleeves": sleeves},
        )
        return response.json()
