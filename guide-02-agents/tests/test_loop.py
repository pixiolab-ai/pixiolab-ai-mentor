"""Chapter 01: the raw loop. The model is a scripted fake (tests/fakes.py)."""
import json

import agent.loop as loop
from tests.fakes import ScriptedClient, response, text, thinking, tool_use


def add(name, args):
    return {"ok": True, "data": args["a"] + args["b"]}


def test_parallel_tool_calls_go_back_in_one_message(monkeypatch):
    fake = ScriptedClient(
        response(thinking(), tool_use("t1", "add", {"a": 1, "b": 2}),
                 tool_use("t2", "add", {"a": 3, "b": 4}), stop="tool_use"),
        response(text("3 and 7."), stop="end_turn"))
    monkeypatch.setattr(loop, "client", fake)
    out = loop.run_agent("add these", tools=[], execute=add)
    assert out == {"status": "end_turn", "answer": "3 and 7."}

    history = fake.requests[1]["messages"]
    assistant, results = history[1], history[2]
    # The whole response goes back unchanged, thinking block included.
    assert [b.type for b in assistant["content"]] == ["thinking", "tool_use",
                                                     "tool_use"]
    # Every result in ONE user message, matched to its call by ID.
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["t1", "t2"]
    assert [json.loads(r["content"])["data"] for r in results["content"]] == [3, 7]
    assert not any(r["is_error"] for r in results["content"])


def test_tool_errors_are_results_not_exceptions(monkeypatch):
    fake = ScriptedClient(
        response(tool_use("t1", "lookup", {}), stop="tool_use"),
        response(text("I couldn't find it."), stop="end_turn"))
    monkeypatch.setattr(loop, "client", fake)
    loop.run_agent("find it", [], lambda n, a: {"ok": False, "error": "nope"})
    result = fake.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True and "nope" in result["content"]


def test_step_cap_stops_a_loop_that_never_answers(monkeypatch):
    forever = [response(tool_use(f"t{i}", "search", {}), stop="tool_use")
               for i in range(3)]
    monkeypatch.setattr(loop, "client", ScriptedClient(*forever))
    out = loop.run_agent("t", [], lambda n, a: {"ok": True}, max_steps=3)
    assert out == {"status": "max_steps", "answer": None}


def test_refusal_or_cutoff_is_reported_with_its_stop_reason(monkeypatch):
    monkeypatch.setattr(loop, "client",
                        ScriptedClient(response(stop="refusal")))
    out = loop.run_agent("t", [], add)
    assert out == {"status": "refusal", "answer": ""}
