"""Chapter 11: the MCP server, called in-process and over stdio."""
import asyncio

from mcp import Client

from servers.orders import mcp


def test_type_hints_and_docstring_become_the_tool_definition():
    async def listing():
        async with Client(mcp) as client:  # in-process, no subprocess
            return await client.list_tools()

    (tool,) = asyncio.run(listing()).tools
    assert tool.name == "lookup_order"
    assert "ORD-12345678" in tool.description
    assert tool.input_schema["properties"]["order_id"]["type"] == "string"
    assert tool.input_schema["required"] == ["order_id"]


def test_calling_the_tool_returns_its_result():
    async def call():
        async with Client(mcp) as client:
            return await client.call_tool("lookup_order",
                                          {"order_id": "ORD-12345678"})

    result = asyncio.run(call())
    assert not result.is_error
    assert '"status": "shipped"' in result.content[0].text


def test_the_stdio_example_launches_the_server_as_a_subprocess():
    from examples import mcp_client
    text = asyncio.run(mcp_client.main())
    assert "ORD-12345678" in text
