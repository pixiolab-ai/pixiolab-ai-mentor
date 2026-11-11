"""Chapter 11: talk to servers/orders.py the way an MCP host would.

The client launches the server as a subprocess and speaks MCP over its
stdin and stdout, lists the tools (the type hints became the input schema,
the docstring the description), then calls one. No API key needed.

Run from guide-02-agents/:  python -m examples.mcp_client
"""
import asyncio
import sys

from mcp import Client, StdioServerParameters


async def main() -> str:
    server = StdioServerParameters(command=sys.executable,
                                   args=["servers/orders.py"])
    async with Client(server) as client:
        listing = await client.list_tools()
        for tool in listing.tools:
            print(f"{tool.name}: {tool.description}\n  {tool.input_schema}")
        result = await client.call_tool("lookup_order",
                                        {"order_id": "ORD-12345678"})
        text = result.content[0].text
        print(f"result (is_error={result.is_error}): {text}")
        return text


if __name__ == "__main__":
    asyncio.run(main())
