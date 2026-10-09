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
        leverage: float = 1.0,
        target_vol: float = 0.0,
        oos_days: int = 90,
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Backtest one portfolio config against the live meta-model.

        The simulator trades the meta-model's published rankings as a
        long/short portfolio with your chosen construction (cohort sizes,
        cadence, optimizer, fees, funding). Returns stats split into
        in-sample (`is_stats`) and out-of-sample (`oos_stats`, the final
        `oos_days`), a config_token, and a web_url the user can open on
        crowdcent.com.

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
        - impact_book: 0-1e9 USD gross (0 = off). Contender.
        Sizing is NOT a config knob: pass `leverage` (0.25-3.0 gross as a
        multiple of equity, default 1.0) and `target_vol` (0-0.5 annualized,
        0 = off, adapts the multiple under that leverage ceiling, never
        above it) as their own arguments; both Contender. Gross past 6x
        equity is liquidated (stats.liquidated_on).
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
            leverage: Gross book as a multiple of equity (default 1.0).
            target_vol: Annualized vol target under `leverage` (0 = off).
            oos_days: Out-of-sample period, the final N days (default 90, 0 =
                none); `is_stats` covers the days before, `oos_stats` the
                test, and `split` echoes the cut.
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
            leverage=leverage,
            target_vol=target_vol,
            oos_days=oos_days,
        )

    @mcp.tool
    def sweep_simulations(
        config: Dict[str, Any],
        sweep: Dict[str, List[Any]],
        oos_days: int = 90,
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Grid-search up to your tier's budget (96 configs at Contender,
        24 below) of portfolio constructions in one call.

        IMPORTANT: choose candidates on in-sample numbers (`is_stats`),
        prefer settings whose neighbors also perform, and read out-of-sample
        numbers (`oos_stats`) only after choosing: picking on the
        out-of-sample period makes it in-sample.

        Args:
            config: Base configuration; swept knobs override it.
            sweep: Sweepable knob -> list of values, e.g.
                {"n_long": [5, 10, 20], "rebalance_days": ["5t", "10t"]}.
                Sweepable: n_long, n_short, rebalance_days, optimizer, lw,
                fee_bps, funding, lag, risk_lookback (Challenger),
                impact_book (Contender), carry (Centurion; only
                carry-aware optimizers such as min_var read it), hedge,
                rank_by, and the factor-lens knobs. Continuous knobs take
                any list of values inside the min/max the capabilities
                listing reports (they snap to its step). Over budget or
                above tier fails with an error that says exactly what is
                allowed.
            oos_days: Out-of-sample period, the final N days (default 90, 0 =
                none); `is_stats` covers the days before, `oos_stats` the
                test, and `split` echoes the cut.
        """
        return client_for(challenge_slug).run_sweep(config, sweep, oos_days=oos_days)

    @mcp.tool
    def blend_simulations(
        sleeves: List[Dict[str, Any]],
        leverage: float = 1.0,
        target_vol: float = 0.0,
        oos_days: int = 90,
        challenge_slug: str = DEFAULT_CHALLENGE,
    ) -> Dict[str, Any]:
        """Blend up to 5-50 weighted sleeves (tier-capped: 5 Challenger, 10
        Contender, 25 Centurion, 50 Sovereign) into one ensemble
        book; returns blend stats plus the sleeve correlation matrix.
        Sizing is the blend's, never a sleeve's: weights shape the blend,
        and the netted book is sized once by leverage / target_vol.

        Args:
            sleeves: [{"config": {...} or "config_token": "...",
                "weight": 1.0, "label": "fast"}, ...]. Low correlation
                between sleeves is what makes a blend worth deploying.
            leverage: Gross book as a multiple of equity (default 1.0).
            target_vol: Annualized vol target under `leverage` (0 = off).
            oos_days: Out-of-sample period, the final N days (default 90, 0 =
                none); `is_stats` covers the days before, `oos_stats` the
                test, and `split` echoes the cut.
        """
        return client_for(challenge_slug).run_blend(
            sleeves,
            leverage=leverage,
            target_vol=target_vol,
            oos_days=oos_days,
        )
