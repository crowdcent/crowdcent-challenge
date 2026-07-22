# Live Trading

The third step of the loop: **Predict → Simulate → Deploy**. Once a construction survives the [Simulator](simulator.md), you can run it live on Hyperliquid through CrowdCent by setting a **mandate**. Live Trading is in staff preview until Trading GA.

You don't place trades on CrowdCent, but you set a mandate, and the execution engine runs it on your Hyperliquid portfolio against the meta-model.

## The Mandate

A mandate is two things:

- **Strategy sleeves**: one or more weighted Simulator configurations. Sleeves use the exact simulation vocabulary, so you deploy precisely what you backtested, by passing a config or the `config_token` from a winning sweep cell.
- **Execution policy**: how the target book becomes orders. Order type (market, limit, post-only ALO, or TWAP), target leverage, slippage tolerance, protective stop-loss/take-profit, and a daily schedule window.

## Custody never moves

The design is fully non-custodial:

- You approve a **trade-only agent key** by signing on-site with your own wallet. Agent keys can rebalance but can never withdraw, and you can revoke them at any time.
- Funds stay in your own Hyperliquid account. CrowdCent stores the agent key encrypted and never your master key.

## The execution process

Each day after the meta-model's rankings publish, your sleeves resolve to a blended **target book**. From there:

1. **Preview.** A rebalance plan is computed and shown first: trades, turnover, estimated fees. Nothing executes.
2. **Confirm.** Executing requires the preview's `plan_hash`, valid for 10 minutes. No fresh preview, no execution.
3. **Execute.** Orders go out under your policy (TWAP slices, ALO resting orders, protective stops). Every fill lands in the order blotter and every run in the audit trail.

Scheduled mode runs this loop for you inside your daily window. You can **pause** the mandate at any time, killing risk is never gated, while resuming requires a trade-enabled key. **Flatten** (close every position) follows the same preview-then-confirm flow.

## Where you can drive it

The same mandate, gates, and audit trail are available from three surfaces:

- **The site**: the Trading tab (mandate form, target book, Orders panel).
- **Python**: `client.set_mandate(...)`, `preview_rebalance()`, `execute_rebalance(plan_hash)`, `pause_trading()`, see the [API reference](api-reference/python.md).
- **AI assistants**: the same tools over [MCP](ai-agents-mcp.md), with the confirm step held by you in conversation.

The Python client and assistant tools default to **testnet**; mainnet is always an explicit opt-in, and the server accepts no ambiguity about which network you mean. Every cap and consent check binds server-side no matter which surface you use.

## Access

Live Trading is in staff preview until Trading GA. When it opens, two switches govern API access: your account's OMS access, and a per-key **"Allow live trading"** toggle, so a key that only reads data can never trade.

Until then, [cc-liquid](https://github.com/crowdcent/cc-liquid), our open-source CLI rebalancer, remains the self-custody way to run the meta-model from your own machine.

!!! warning "Trading Disclaimer"
    Not financial, investment, or trading advice. Perpetual futures are leveraged instruments and you can lose your entire margin. Simulated performance is not indicative of future results. See the full [disclaimer](disclaimer.md).
