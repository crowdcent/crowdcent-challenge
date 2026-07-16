# From Predictions to Portfolios

You built a model, submitted predictions, and earned [CC Points](points-system.md). This page covers what happens next: turning the [meta-model](hyperliquid-ranking.md#meta-model) into a portfolio you can study, stress-test, and eventually deploy.

The loop looks like this:

1. **Predict.** Build a model and submit rankings during inference periods.
2. **Simulate.** Backtest the live meta-model as a long/short portfolio in the [Simulator](https://crowdcent.com/challenge/hyperliquid-ranking/meta-model/simulation/).
3. **Deploy.** Run a simulated strategy live on Hyperliquid through CrowdCent (currently in staff preview).

## The Simulator

The [Simulator](https://crowdcent.com/challenge/hyperliquid-ranking/meta-model/simulation/) turns the meta-model's daily rankings into long/short perpetual-futures portfolios over real Hyperliquid market data. No code required, and you can try it without logging in.

You control the strategy at a high level:

- **Selection and weighting**: how many names per leg and how they're sized (equal weight, inverse-vol, HRP, and more at higher tiers)
- **Cadence**: rebalance frequency, including tranched "rolling vintage" modes that remove rebalance-timing luck
- **Risk**: volatility targeting and liquidity floors
- **Realism stress**: fees, market impact at size, and signal lag, so you can see how much edge survives real-world frictions

Every backtest reports an out-of-sample holdout alongside the full-period result.

### Sweeps and sleeves

Instead of testing one configuration at a time, **parameter sweeps** run a grid of configurations and render the results as a heatmap. Read plateaus, not peaks: a lone bright cell is luck, a bright region is structure. The interface is built to surface robust parameter regions rather than overfit point estimates.

**Sleeves** let you blend several configurations into one ensemble strategy. It's the same diversification logic the Challenge applies to models, applied to portfolios.

## Tier Unlocks

Simulator capability scales with [CC Points](points-system.md), earned through predictive skill:

| Tier | Simulator unlocks |
|:---|:---|
| Everyone | Full-realism backtesting on 90-day delayed meta-model data |
| **Challenger** (100+) | Real-time meta-model, Inverse-Vol & HRP weighting, parameter sweeps |
| **Contender** (500+) | Risk & capacity controls, full sweep & blend budgets |
| **Centurion** (1,500+) | Classified alpha controls |

## Live Trading

Deploying a simulated strategy to Hyperliquid through CrowdCent is running in a staff preview and will open beyond staff over time. The design is non-custodial: you approve a trade-only agent key that can rebalance but never withdraw, funds stay in your own account, and the key is revocable at any time.

Until then, [cc-liquid](https://github.com/crowdcent/cc-liquid), our open-source CLI rebalancer, remains the self-custody way to run the meta-model from your own machine. Strategy research now lives in the Simulator, and the CLI's MIT source stays available.

!!! warning "Simulator Disclaimer"
    Simulations are provided for informational and educational purposes only. Not financial, investment, or trading advice. Simulated performance is not indicative of future results. Perpetual futures are leveraged instruments and you can lose your entire margin. See the full [disclaimer](disclaimer.md).
