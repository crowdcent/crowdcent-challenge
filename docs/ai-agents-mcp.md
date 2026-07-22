# Using AI Agents

The `crowdcent-challenge` package ships with a built-in Model Context Protocol (MCP) server, so AI assistants like Claude, Cursor, and Codex can work with the CrowdCent Challenge in natural language: download data, submit predictions, check performance, backtest portfolio constructions on the meta-model, and manage live trading (a staff preview until Trading GA).

## Setup

**1. Get an API key.** Create an account at [crowdcent.com](https://crowdcent.com) and generate a key in your [profile settings](https://crowdcent.com/profile/settings/).

**2. Connect.** There are two ways: the hosted server, which needs nothing installed, or a local server run with `uvx`.

=== "Hosted (nothing to install)"

    Add `https://mcp.crowdcent.com/mcp` to any client that supports remote MCP servers over HTTP, using your API key as a bearer token.

    Claude Code:

    ```bash
    claude mcp add --transport http crowdcent https://mcp.crowdcent.com/mcp \
      --header "Authorization: Bearer YOUR_API_KEY"
    ```

    Cursor / VS Code (`mcp.json`):

    ```json
    {
      "mcpServers": {
        "crowdcent": {
          "url": "https://mcp.crowdcent.com/mcp",
          "headers": {
            "Authorization": "Bearer YOUR_API_KEY"
          }
        }
      }
    }
    ```

    The hosted server is a stateless pass-through to the CrowdCent API: it stores nothing, and your key simply authenticates each request the same way the Python client does. Instead of writing files to disk, it hands your assistant signed download URLs (`get_training_dataset_url`, `get_inference_data_url`, `get_meta_model_url`), which agents with a terminal or fetch tool can use directly.

=== "Local (one line, no clone)"

    If you prefer running locally, or want the file-download and file-submission tools, configure your client with a single `uvx` command:

    ```json
    {
      "mcpServers": {
        "crowdcent": {
          "command": "uvx",
          "args": ["--from", "crowdcent-challenge[mcp]", "crowdcent-mcp"],
          "env": {"CROWDCENT_API_KEY": "your_api_key_here"}
        }
      }
    }
    ```

    That is the whole setup: no repository to clone, no absolute paths, and the server always matches the client library it ships with.

**3. Say hello.** Confirm the connection with a first ask:

```
"List the CrowdCent challenges and tell me about the current inference period."
```

## What to ask

You don't need to know the tool names, just describe what you want. Some starting points, roughly in the order of a CrowdCent journey:

**Explore the challenge**

```
"What does the hyperliquid-ranking challenge involve, and what would I submit?"
"Show my recent submissions and how they scored."
```

**Build and submit a model.** Your assistant can run the whole loop: pull data, write and train a model in your workspace, and submit the predictions, all in one conversation.

```
"Download the latest CrowdCent training data, explore it, and train a
baseline ranking model. Predict the current inference period and show me
the predictions before submitting."
```

Coding agents like Claude Code and Cursor handle this end to end. Chat-only clients can still fetch data via the signed-URL tools and submit inline from a dataframe.

**Backtest on the meta-model.** The simulation tools answer "how would trading the community's aggregated signal have performed?" with the same engine, tier gates, and honesty stance as the site's Simulation tab. Sweep configurations, blend sleeves, and compare in-sample against out-of-sample.

```
"Backtest 10 long / 10 short equal weight on the meta-model, rebalanced
every 10 days with funding on, and give me the link to view it on the site."

"Sweep leg size 5/10/20 and cadence 1/5/10 days on the meta-model, and tell
me where the stable region is, in-sample and out."

"Blend a fast small-book sleeve with a slower HRP sleeve 60/40 and show the
combined out-of-sample stats and correlation."
```

The sweep tool's own guidance applies: read plateaus, not peaks. A lone bright cell is luck; a bright region is structure. Every result includes a `web_url` deep link, so anything your assistant finds can be opened and inspected on crowdcent.com.

**Check in each morning** (trading-enabled accounts):

```
"Give me my CrowdCent morning briefing."
```

## Built-in workflows

The server ships two prompts, packaged versions of the asks above that encode the recommended workflow. In Claude Code they appear as slash commands (`/mcp__crowdcent__sweep_and_summarize`, `/mcp__crowdcent__morning_briefing`); other clients surface them in their prompt picker.

- **`sweep_and_summarize`**: runs a sweep, tables in-sample and out-of-sample Sharpe, identifies the stable plateau rather than the single best cell, recommends one configuration from inside it, and hands you the site deep link.
- **`morning_briefing`**: pulls your recent scores, account state, last rebalance results, and working orders, then narrates what needs your attention. Built for trading-enabled accounts; anyone can use it for the scoring summary.

## Live trading (staff preview until Trading GA)

For trading-enabled accounts, the same tools that power the site's Trading tab: inspect the mandate and target book, preview a rebalance, execute it after your explicit confirmation, flatten, pause, and read the run and order audit trail. The concepts (mandates, custody, the consent flow) are covered in [Live Trading](live-trading.md).

```
"Set the mandate to the sweep config we just picked, preview the rebalance
on testnet, and walk me through the plan before executing anything."
```

Trading tools appear when your API key has live trading enabled (Settings → "Allow live trading") and your account has OMS access (staff preview until Trading GA). That is the same locally and on the hosted server. Execution is always two-step: the assistant previews a plan, shows it to you, and needs the preview's `plan_hash` (valid for 10 minutes) to execute. Every cap and gate is enforced server-side, and custody never moves: agent keys are trade-only, never-withdraw, and stay encrypted on CrowdCent's side. The MCP server holds nothing but your API key.

## Troubleshooting

- **"API key not provided"**: set `CROWDCENT_API_KEY` in the server's `env` block (local) or the `Authorization` header (hosted).
- **Trading tools missing**: enable "Allow live trading" on your key in the trading tab. Trading is in staff preview until Trading GA — without OMS access on your account, tools stay hidden and API calls would fail anyway.
- **A knob didn't take effect**: simulation knobs above your points tier are clamped to your tier, and the response's `locked` list names which ones. Error messages state the tier and points needed to unlock.
- **Submission format**: predictions need the challenge's required columns (for `hyperliquid-ranking`: `id`, `pred_10d`, `pred_30d`), and submissions are only open during the challenge's submission window.
