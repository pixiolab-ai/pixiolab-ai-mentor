from mcp.server.mcpserver import MCPServer

mcp = MCPServer("orders")

@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up one order by ID (format ORD-12345678). Read-only."""
    return {"order_id": order_id, "status": "shipped"}  # call your API

if __name__ == "__main__":
    mcp.run(transport="stdio")    # "streamable-http" for a remote server
