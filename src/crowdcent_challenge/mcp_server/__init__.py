"""CrowdCent MCP server, packaged with the client (`crowdcent-challenge[mcp]`).

Nothing heavy is imported here: the base install must never pull fastmcp.
Run it with the `crowdcent-mcp` console script (stdio), or serve
`crowdcent_challenge.mcp_server.app:http_app` under uvicorn (hosted).
"""
