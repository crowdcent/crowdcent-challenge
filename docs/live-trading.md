# Live Trading

Live trading allows you to deploy backtested [Simulator](simulator.md) strategies directly to Hyperliquid perpetual markets against daily meta-model updates.

Hosted live trading is available to **Challenger-tier participants** (100+ CC Points) with a submission in the last 30 days.

## Mandate architecture

Live trading operates under a **mandate**, which defines the portfolio composition and how target allocations are converted into exchange orders:

- **Strategy sleeves**: One or more weighted Simulator configurations. Sleeves use the exact simulation parameter vocabulary (or `config_token` values from backtest runs).
- **Execution policy**: Order execution type (`market`, `limit`, post-only `alo`, or `twap` with duration), target leverage, maximum slippage tolerance, protective stop-loss/take-profit triggers, and scheduled daily execution windows.
- **Sizing**: The mandate sizes the book once. Sleeves are run at natural gross and netted; `target_leverage` is the multiple of account value deployed (server-capped at 3x), and the mandate's `target_vol` (0 = off) adapts that multiple under the leverage ceiling. Sizing knobs inside a sleeve config are ignored. The target book endpoint returns natural-gross weights and the one `gross_multiplier` the planner applies. The Trading tab's simulator mirror runs under the same sizing, liquidation line included.

## Non-custodial security

Hosted live trading uses a non-custodial design:

- **Trade-only agent keys**: You authorize a scoped API agent key on Hyperliquid by signing with your wallet. Agent keys can place and cancel orders, but cannot transfer collateral or withdraw funds.
- **Revocation**: You can revoke agent keys at any time directly through Hyperliquid or the CrowdCent dashboard.
- **Master key protection**: CrowdCent never stores or handles your master wallet private keys.

## Two-step execution workflow

Each day after the meta-model publishes new rankings, your strategy sleeves resolve into a target portfolio. Execution uses a two-step confirmation workflow:

1. **Preview (`preview_rebalance`)**: Calculates the required position adjustments, turnover, and estimated fees against current balances and target allocations. Returns a signed `plan_hash` valid for 10 minutes without placing orders.
2. **Execute (`execute_rebalance`)**: Consumes the `plan_hash` to authorize order submission. The execution engine recalculates target orders against live order books while enforcing account caps and slippage limits.

When scheduled execution is enabled, this loop runs automatically within your configured daily time window.

## Safety and emergency controls

- **Testnet by default**: The SDK, REST API, and MCP tools default to `network="testnet"`. Production trading requires explicitly passing `network="mainnet"`.
- **Immediate pause (`pause_trading`)**: Instantly disables scheduled execution. Can be triggered with any valid API key.
- **Resume trading (`resume_trading`)**: Re-enables scheduled execution (requires an API key with live trading permissions).
- **Position liquidation (`flatten`)**: Closes all active positions using a dedicated two-step preview and execution flow (`flatten(preview=True)` followed by `flatten(plan_hash=...)`).

## Python quickstart

```python
from crowdcent_challenge import ChallengeClient

client = ChallengeClient("hyperliquid-ranking")

# 1. Configure the mandate with strategy sleeves and execution settings
mandate = client.set_mandate(
    {
        "sleeves": [
            {
                "config": {
                    "n_long": 10,
                    "n_short": 10,
                    "optimizer": "inv_vol",
                    "rebalance_days": "10t",
                    "include_funding": True,
                },
                "weight": 1.0,
                "label": "Primary Model",
            }
        ],
        "order_type": "twap",
        "twap_minutes": 15,
        "target_leverage": 1.0,
        "schedule_enabled": True,
        "schedule_at_time": "14:00",
    },
    network="testnet",
)

# 2. Preview the rebalance plan (dry run)
preview = client.preview_rebalance(network="testnet")
print(f"Planned trades: {len(preview['trades'])}")
print(f"Estimated turnover: ${preview['turnover']:,.2f}")
print(f"Plan hash: {preview['plan_hash']}")

# 3. Confirm and execute using the plan hash within 10 minutes
execution = client.execute_rebalance(
    plan_hash=preview["plan_hash"],
    network="testnet",
)
print("Execution status:", execution["status"])

# 4. Emergency pause
# client.pause_trading(network="testnet")
```

## Access requirements

Live trading requires:

1. **Challenger tier** or above (100+ CC Points).
2. An active submission in the last 30 days.
3. An API key with the **Allow live trading** permission enabled in your [profile settings](https://crowdcent.com/profile/settings/).

!!! warning "Trading Disclaimer"
    Not financial, investment, or trading advice. Perpetual futures are leveraged instruments and you can lose your entire margin. Simulated performance is not indicative of future results. See the full [disclaimer](disclaimer.md).
