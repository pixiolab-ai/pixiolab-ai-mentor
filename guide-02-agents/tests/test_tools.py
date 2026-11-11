"""Chapters 02 and 09: the validating registry and a well-designed definition."""
import json
import re

import agent.tools as tools
from agent.tools.lookup_order import LOOKUP_ORDER


def test_definitions_come_from_the_pydantic_schema():
    (definition,) = tools.tool_definitions()
    assert definition["name"] == "search_docs"
    props = definition["input_schema"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["max_results"]["maximum"] == 10


def test_unknown_tool_lists_the_real_ones():
    out = tools.execute("search_web", {})
    assert not out["ok"] and "Available: search_docs" in out["error"]


def test_invalid_arguments_are_returned_to_the_model():
    out = tools.execute("search_docs", {"max_results": 50})
    assert not out["ok"] and out["error"].startswith("Invalid arguments")
    json.dumps(out)  # serializable, so it can go back as a tool_result


def test_a_tool_that_raises_becomes_an_error_result():
    out = tools.execute("search_docs", {"query": "refunds"})
    assert out == {"ok": False, "error": "NotImplementedError: call your "
                                         "search backend here"}


def test_valid_call_runs_with_defaults_applied(monkeypatch):
    monkeypatch.setitem(tools.TOOLS["search_docs"], "fn",
                        lambda query, max_results: [query, max_results])
    assert tools.execute("search_docs", {"query": "refunds"}) == {
        "ok": True, "data": ["refunds", 5]}


def test_lookup_order_definition_is_strict_json_schema():
    schema = LOOKUP_ORDER["input_schema"]
    assert json.loads(json.dumps(LOOKUP_ORDER)) == LOOKUP_ORDER
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["order_id"]
    pattern = schema["properties"]["order_id"]["pattern"]
    assert re.fullmatch(pattern, "ORD-12345678")
    assert not re.fullmatch(pattern, "12345678")
