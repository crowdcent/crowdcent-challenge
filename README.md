<div align="center">
  <img src="docs/overrides/assets/images/cc_logo.png" alt="CrowdCent Logo" width="200">
  
  <h1>CrowdCent Challenge</h1>
  
  <p>Open data science competitions for ML engineers and data scientists</p>
  
  [![PyPI](https://img.shields.io/pypi/v/crowdcent-challenge?style=flat-square&color=blue)](https://pypi.org/project/crowdcent-challenge/)
  [![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/github/license/crowdcent/crowdcent-challenge?style=flat-square)](LICENSE)
  [![Downloads](https://img.shields.io/pypi/dm/crowdcent-challenge?style=flat-square&color=green)](https://pypi.org/project/crowdcent-challenge/)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/v6ZSGuTbQS)
  [![Docs](https://img.shields.io/badge/docs-crowdcent.com-orange?style=flat-square)](https://docs.crowdcent.com)
  
  <br>
  
  [![Get Started](https://img.shields.io/badge/Get%20Started-→-brightgreen?style=for-the-badge)](https://docs.crowdcent.com/getting-started/)
  [![View Challenges](https://img.shields.io/badge/View%20Challenges-→-blue?style=for-the-badge)](https://crowdcent.com/challenge)
  
</div>

---

The CrowdCent Challenge is an open data science competition designed for machine learning engineers, data scientists, AI agents, and other technical professionals to hone their skills in a real-world setting.

## What is CrowdCent?
CrowdCent is on a mission to decentralize investment management by changing the way investment funds make decisions and allocate capital. We are the machine learning and coordination layer for online investment communities looking to turn their data into [actionable, investable portfolios](https://docs.crowdcent.com/simulator/).

## Installation

[![uv](https://img.shields.io/badge/uv-Recommended-6B57FF?style=flat-square)](https://github.com/astral-sh/uv)
[![pip](https://img.shields.io/badge/pip-Compatible-blue?style=flat-square)](https://pip.pypa.io/)

### Using uv (Recommended)
```bash
uv add crowdcent-challenge
```

### Using pip
```bash
pip install crowdcent-challenge
```

## Quick Start

1. **Get an API Key**: Generate your key from your [profile settings](https://crowdcent.com/profile/settings/)
2. **Set up authentication**:
   ```bash
   export CROWDCENT_API_KEY=your_api_key_here
   # or create a .env file with: CROWDCENT_API_KEY=your_api_key_here
   ```
3. **Start competing**:
   ```python
   from crowdcent_challenge import ChallengeClient
   
   # Initialize client for a challenge
   client = ChallengeClient(challenge_slug="hyperliquid-ranking")
   
   # Download training data
   client.download_training_dataset("latest", "training_data.parquet")
   
   # Download inference data
   client.download_inference_data("current", "inference_data.parquet")
   
   # Submit predictions
   client.submit_predictions(file_path="predictions.parquet")

   # Or test a new model with an experimental submission + private note
   # (requires another slot to have a non-experimental submission)
   client.submit_predictions(
       file_path="predictions.parquet",
       slot=2,
       is_experimental=True,
       notes="2-layer transformer w/ sector embeddings",
   )
   ```

## Available Challenges

- **[Hyperliquid Ranking](https://crowdcent.com/challenge/hyperliquid-ranking)**: Rank crypto assets on Hyperliquid by expected relative returns
[![Hyperliquid Challenge](https://img.shields.io/badge/Challenge-Hyperliquid%20Ranking-blue?style=flat-square)](https://crowdcent.com/challenge/hyperliquid-ranking)

- **Equity NLP**: Coming soon!
[![Equity NLP](https://img.shields.io/badge/Challenge-Equity%20NLP-gray?style=flat-square)](https://crowdcent.com/challenge)

## CLI Usage

The package includes a command-line interface:
```bash
# List all challenges
crowdcent list-challenges

# Set default challenge
crowdcent set-default-challenge hyperliquid-ranking

# Download data
crowdcent download-training-data latest -o training.parquet
crowdcent download-inference-data current -o inference.parquet

# Submit predictions
crowdcent submit predictions.parquet

# Submit an experimental prediction with a private note
crowdcent submit predictions.parquet --slot 2 --experimental \
    --notes "2-layer transformer w/ sector embeddings"
```

**Documentation**: [docs.crowdcent.com](https://docs.crowdcent.com)

## AI Agents / MCP Server

This package ships with a built-in Model Context Protocol (MCP) server, so assistants like Claude and Cursor can download data, submit predictions, backtest strategies on the meta-model, and check your performance in natural language. Connect the hosted server at `https://mcp.crowdcent.com/mcp` with your API key, or run it locally with one line:

```json
{"mcpServers": {"crowdcent": {
  "command": "uvx",
  "args": ["--from", "crowdcent-challenge[mcp]", "crowdcent-mcp"],
  "env": {"CROWDCENT_API_KEY": "your_api_key_here"}
}}}
```

See the [AI Agents guide](https://docs.crowdcent.com/ai-agents-mcp/) for hosted setup, the simulation tools, and the trading preview.

## Contributing

Contributions are welcome! The `crowdcent-challenge` client library and documentation are open source.

See our [contributing guidelines](https://docs.crowdcent.com/contributing/) for details on:
- Forking and cloning the repository
- Setting up development environment
- Making changes and submitting PRs

## Have Questions?

[![Documentation](https://img.shields.io/badge/Documentation-docs.crowdcent.com-orange?style=for-the-badge)](https://docs.crowdcent.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/v6ZSGuTbQS)
[![Email](https://img.shields.io/badge/Email-info@crowdcent.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:info@crowdcent.com)