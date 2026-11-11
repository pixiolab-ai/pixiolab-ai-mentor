"""The examples run end to end: offline ones as they are, model-backed ones
with the scripted fake client in place of the real one."""
import json

import agent.loop as loop
import agent.router as router
from tests.fakes import (ScriptedClient, import_state_in, response, text,
                         tool_use)


def test_support_agent_validates_and_runs_both_tools(monkeypatch):
    from examples import support_agent
    fake = ScriptedClient(
        response(tool_use("t1", "lookup_order", {"order_id": "12345678"}),
                 stop="tool_use"),               # bad format: error result
        response(tool_use("t2", "lookup_order", {"order_id": "ORD-12345678"}),
                 tool_use("t3", "search_docs", {"query": "track package"}),
                 stop="tool_use"),
        response(text("It shipped on November 3."), stop="end_turn"))
    monkeypatch.setattr(loop, "client", fake)
    out = support_agent.main(["Where is my order 12345678?"])
    assert out == {"status": "end_turn", "answer": "It shipped on November 3."}

    first = fake.requests[1]["messages"][2]["content"][0]
    assert first["is_error"] and "Invalid arguments" in first["content"]
    order, docs = fake.requests[2]["messages"][4]["content"]
    assert json.loads(order["content"])["data"]["status"] == "shipped"
    assert json.loads(docs["content"])["data"][0]["title"] == "Track a package"
    names = [t["name"] for t in fake.requests[0]["tools"]]
    assert names == ["search_docs", "lookup_order"]


def test_budgeted_agent_stops_repeats_and_flags_the_limit(monkeypatch):
    from agent.budget import Budget
    from agent.tools import execute
    from examples import budgeted_agent, support_agent
    same_call = tool_use("t", "search_docs", {"query": "returns"})
    fake = ScriptedClient(*[response(same_call, stop="tool_use")] * 4)
    monkeypatch.setattr(loop, "client", fake)
    out = budgeted_agent.run_budgeted_agent(
        "loop forever", support_agent.register_tools(), execute,
        Budget(max_steps=4))
    assert out["status"] == "budget_exceeded:steps"
    assert out["used"]["tokens"] == 4 * 120
    third_result = fake.requests[3]["messages"][-1]["content"][0]
    assert third_result["is_error"] and "already ran" in third_result["content"]


def test_router_demo_picks_a_threshold_from_its_labels(monkeypatch):
    from examples import router_demo
    labels = dict(router_demo.LABELED)

    def fake_route(request):
        category = labels.get(request, "support")
        return router.Route(category=category, confidence=0.9,
                            order_id="ORD-55510001")

    monkeypatch.setattr(router_demo, "route", fake_route)
    monkeypatch.setattr(router, "route", fake_route)  # used by handle()
    assert router_demo.main() == 0.0  # every fake route is right


def test_offline_examples_run(tmp_path, monkeypatch, capsys):
    from examples import planning_demo, reflection_demo, team_demo
    assert planning_demo.main()["status"] == "done"
    reflected = reflection_demo.main()
    assert (reflected["status"], reflected["attempts"]) == ("passed", 2)
    assert "re.sub" in reflected["draft"]  # the second draft passed its tests
    assert team_demo.main()["missing"] == ["writer"]

    state = import_state_in(tmp_path, monkeypatch)  # runs.db goes to tmp
    from examples import resume_demo
    finished = resume_demo.main()
    assert finished.step == 4 and finished.final == "reply sent"
    state.db.close()
    assert "crash: worker died before step 2" in capsys.readouterr().out
