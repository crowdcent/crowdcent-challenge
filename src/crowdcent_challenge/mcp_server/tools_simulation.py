"""Simulation tools: backtest, sweep, and blend portfolio constructions on
the live meta-model — the same engine and tier gates as the site's
Simulation tab. Pure API pass-throughs; client exceptions propagate and
fastmcp returns their messages verbatim as tool errors."""

from __future__ import annotations

from typing import Any, Dict, List

from .runtime import DEFAULT_CHALLENGE, client_for


def register_simulation_tools(mcp) -> None:
    @mcp.tool
    def run_simulation(
        config: Dict[str, Any],
        include_curve: bool = False,
        include_holdings: bool = False,
        benchmark_trials: int = 0,
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Backtest one portfolio config against the live meta-model.

        The simulator trades the meta-model's published rankings as a
        long/short portfolio with your chosen construction (cohort sizes,
        cadence, optimizer, fees, funding). Returns stats with an
        in-sample/out-of-sample split, a config_token, and a web_url the
        user can open on crowdcent.com.

        Config knobs (all optional; omitted knobs use the site's defaults:
        pred_30d, 40/40, "10t", a $1.5M open-interest floor, 3.5 bps fees
        and funding on, and the best optimizer the key's tier unlocks —
        min_var at Contender, with carry_penalty 0.25 at Centurion):
        - rank_by: "pred_10d" | "pred_30d" | "blend" (Centurion tier)
        - n_long / n_short: names per leg, 1-100 each
        - rebalance_days: "1"|"5"|"10"|"30", or tranched "5t"|"10t"|"30t"
        - optimizer: "equal" | "inv_vol"/"hrp" (Challenger); higher tiers
          unlock covariance optimizers (Contender) and conviction
          sizing (Centurion) — the capabilities listing names the
          keys your tier unlocks
        - fee_bps: 0-20 bps per side (default 3.5, Hyperliquid taker);
          include_funding: bool (default true)
        - signal_lag: 0-14 days; risk_lookback: 10-365 days in 5d steps,
          0 = the optimizer's own window (Challenger)
        - leverage: 0.25-3.0 gross as a multiple of equity (default 1.0);
          target_vol: 0-0.5 annualized (0 = off) adapts the multiple under
          that leverage ceiling, never above it. Gross past 6x equity is
          liquidated (stats.liquidated_on). impact_book: 0-1e9 USD gross
          (0 = off). All Contender.
        - hedge_btc: bool and carry_penalty: 0-1.0 (Centurion);
          lw_shrinkage: bool. Continuous knobs snap to the step the
          capabilities listing reports (min/max/step per knob).
        - min_oi / min_volume / min_trades: liquidity floors (USD, USD, count)
        - start_date: "YYYY-MM-DD"

        Knobs above the user's tier are silently clamped, never an error —
        the response's `locked` list names what was clamped, so just run
        and check it. No capability pre-check call is needed.

        Args:
            config: Knob dict, e.g. {"n_long": 10, "n_short": 10,
                "rebalance_days": "10t", "optimizer": "inv_vol",
                "include_funding": True}.
            include_curve: Also return the full daily equity series.
            include_holdings: Also return the current simulated book.
            benchmark_trials: 0-100 random-ranking portfolios to score the
                signal against.
        """
        include: List[str] = []
        if include_curve:
            include.append("curve")
        if include_holdings:
            include.append("holdings")
        return client_for(challenge_slug).run_simulation(
            config=config,
            include=include or None,
            benchmark_trials=benchmark_trials,
        )

    @mcp.tool
    def sweep_simulations(
        config: Dict[str, Any],
        sweep: Dict[str, List[Any]],
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Grid-search up to your tier's budget (96 configs at Contender,
        24 below) of portfolio constructions in one call.

        IMPORTANT: read plateaus, not peaks — a lone bright cell is luck; a
        bright region is structure. Prefer configs whose neighbors also
        perform, and weight out-of-sample stats (`oos_stats`) over
        in-sample when recommending anything.

        Args:
            config: Base configuration; swept knobs override it.
            sweep: Sweepable knob -> list of values, e.g.
                {"n_long": [5, 10, 20], "rebalance_days": ["5t", "10t"]}.
                Sweepable: n_long, n_short, rebalance_days, optimizer, lw,
                fee_bps, funding, lag, risk_lookback (Challenger),
                leverage, target_vol, impact_book (Contender), carry (Centurion; only
                carry-aware optimizers such as min_var read it), hedge,
                rank_by, and the factor-lens knobs. Continuous knobs take
                any list of values inside the min/max the capabilities
                listing reports (they snap to its step). Over budget or
                above tier fails with an error that says exactly what is
                allowed.
        """
        return client_for(challenge_slug).run_sweep(config, sweep)

    @mcp.tool
    def blend_simulations(
        sleeves: List[Dict[str, Any]],
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Blend up to 3-5 weighted sleeves (tier-capped) into one ensemble
        book; returns blend stats plus the sleeve correlation matrix.

        Args:
            sleeves: [{"config": {...} or "config_token": "...",
                "weight": 1.0, "label": "fast"}, ...]. Low correlation
                between sleeves is what makes a blend worth deploying.
        """
        return client_for(challenge_slug).run_blend(sleeves)
