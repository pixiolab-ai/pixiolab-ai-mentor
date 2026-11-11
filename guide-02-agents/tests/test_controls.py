"""Chapters 08, 10, 12 and 13: budgets, checkpoints, trajectory checks and
idempotency keys."""
import json

import pytest

from agent.budget import Budget
from evals.trajectory import check_trajectory
from tests.fakes import import_state_in
from tools.refund import idempotency_key, issue_refund


# ---------- Chapter 08: budgets ----------

def test_each_limit_names_itself_when_hit():
    steps = Budget(max_steps=2)
    assert steps.exceeded() is None
    steps.charge(tokens=10, cost_usd=0.01)
    steps.charge(tokens=10, cost_usd=0.01)
    assert steps.exceeded() == "steps"

    cost = Budget(max_cost_usd=0.05)
    cost.charge(tokens=1, cost_usd=0.06)
    assert cost.exceeded() == "cost_usd"
    assert Budget(max_seconds=0).exceeded() == "seconds"


def test_the_third_identical_call_is_a_repeat():
    b = Budget()
    assert not b.repeated("search", {"q": "refunds", "k": 5})
    assert not b.repeated("search", {"k": 5, "q": "refunds"})  # same call
    assert b.repeated("search", {"q": "refunds", "k": 5})
    assert not b.repeated("search", {"q": "returns", "k": 5})


# ---------- Chapter 10: state and checkpoints ----------

@pytest.fixture
def state(tmp_path, monkeypatch):
    """agent.state opens runs.db in the working directory at import time,
    so import it fresh inside a temporary directory."""
    module = import_state_in(tmp_path, monkeypatch)
    yield module
    module.db.close()


def test_resume_returns_the_latest_checkpoint(state):
    s = state.AgentState(run_id="r1")
    state.checkpoint(s)
    s.step, s.final = 3, "done"
    state.checkpoint(s)
    resumed = state.resume("r1")
    assert (resumed.step, resumed.final) == (3, "done")
    assert state.resume("no-such-run") is None


def test_old_checkpoints_load_with_defaults_for_new_fields(state):
    v1 = {"schema_version": 1, "run_id": "old", "step": 1}  # no awaiting_human
    state.db.execute("INSERT INTO checkpoints VALUES (?, ?, ?)",
                     ("old", 1, json.dumps(v1)))
    resumed = state.resume("old")
    assert resumed.schema_version == 1 and resumed.awaiting_human is False


# ---------- Chapter 12: trajectory checks ----------

def tool(name, **extra):
    return {"kind": "tool", "name": name, **extra}


@pytest.mark.parametrize("events, problems", [
    ([{"kind": "answer"}], ["answered before retrieving"]),
    ([tool("search_docs"), tool("refund"), {"kind": "answer"}],
     ["refund issued without approval"]),
    ([tool("search_docs"), tool("refund", approved=True), {"kind": "answer"}],
     []),
    ([tool("search_docs")] * 6 + [{"kind": "answer"}],
     ["6 searches (limit 5)"]),
])
def test_trajectory_rules(events, problems):
    assert check_trajectory(events) == problems


# ---------- Chapter 13: idempotency keys ----------

class Payments:
    """A payment API that stores idempotency keys, as real ones do."""

    def __init__(self):
        self.refunds = {}

    def refund(self, order_id, amount_cents, idempotency_key):
        self.refunds.setdefault(idempotency_key, (order_id, amount_cents))
        return {"ok": True, "refund_id": idempotency_key[:12]}


def test_a_retried_step_refunds_once():
    payments = Payments()
    args = {"order_id": "ORD-12345678", "amount_cents": 1500}
    first = issue_refund(payments, "run-1", 4, args)
    retry = issue_refund(payments, "run-1", 4, dict(reversed(args.items())))
    assert first == retry and len(payments.refunds) == 1


def test_a_different_step_or_amount_is_a_different_key():
    args = {"order_id": "ORD-12345678", "amount_cents": 1500}
    key = idempotency_key("run-1", 4, "issue_refund", args)
    assert key != idempotency_key("run-1", 5, "issue_refund", args)
    assert key != idempotency_key("run-1", 4, "issue_refund",
                                  {**args, "amount_cents": 1501})
