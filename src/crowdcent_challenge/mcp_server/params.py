"""Shared argument types and tool annotations for the MCP tools.

The descriptions go into each tool's input schema, so an agent knows what
an argument means without reading the docstring."""

from __future__ import annotations

from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

ChallengeSlug = Annotated[
    str,
    Field(
        description="Challenge slug, such as hyperliquid-ranking. "
        "list_all_challenges gives the slugs."
    ),
]

Network = Annotated[
    str,
    Field(description='Hyperliquid network: "testnet" or "mainnet".'),
]

READ_ONLY = ToolAnnotations(readOnlyHint=True)
