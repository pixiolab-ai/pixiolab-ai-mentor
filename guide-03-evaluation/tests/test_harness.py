"""Chapter 05: the tiny harness, and run_tickets.py end to end with a fake
summarizer in place of the model."""
import json
import runpy
import shutil
from pathlib import Path

import pytest

import app.tickets
from harness import grade, run_suite, summarize

RUN_TICKETS = Path(__file__).resolve().parents[1] / "evals/run_tickets.py"

CASES = [
    {"id": "t1", "input": "charged twice", "slice": {"intent": "billing"},
     "expect": {"category": "billing", "key_facts": ["charged twice"]}},
    {"id": "t2", "input": "reset link expired", "slice": {"intent": "account"},
     "expect": {"category": "account", "key_facts": ["reset link"]}},
    {"id": "t3", "input": "no slice tag", "expect": {"category": "technical",
                                                     "key_facts": ["slow"]}},
]
GRADERS = {
    "right_category": lambda c, o: o["category"] == c["expect"]["category"],
    "key_facts": lambda c, o: all(f in o["summary"]
                                  for f in c["expect"]["key_facts"]),
}


def test_a_grader_that_crashes_fails_its_check():
    checks = grade(GRADERS, CASES[0], {"summary": "no category key"})
    assert checks == {"right_category": False, "key_facts": False}


def test_crashes_are_recorded_failures_and_trials_are_averaged_per_case():
    calls = {"n": 0}

    def flaky_system(text):
        calls["n"] += 1
        if text == "reset link expired" and calls["n"] % 3 == 0:
            raise TimeoutError("model timed out")
        return {"category": "billing" if "charged" in text else "account",
                "summary": f"Customer says: {text}"}

    records = run_suite(CASES, flaky_system, GRADERS, trials=3)
    assert len(records) == 9
    crashed = [r for r in records if r["error"]]
    assert crashed and all(not r["passed"] and r["checks"] == {}
                           for r in crashed)
    summary = summarize(records)
    assert summary["billing"] == (1.0, 1)          # (pass rate, n cases)
    assert summary["account"] == (round(2 / 3, 3), 1)
    assert summary["none"] == (0.0, 1)             # untagged cases
    assert summary["ALL"][1] == 3                  # cases, not trials


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A temporary project folder holding a copy of the dataset."""
    (tmp_path / "evals").mkdir()
    shutil.copy("evals/tickets_v3.jsonl", tmp_path / "evals")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    return tmp_path


def test_run_tickets_writes_every_record_with_its_versions(
        workdir, monkeypatch):
    with open("evals/tickets_v3.jsonl") as f:
        expected = {c["input"]: c["expect"] for c in map(json.loads, f)}

    def perfect(ticket):  # stands in for the model
        expect = expected[ticket]
        return {"category": expect["category"],
                "summary": "; ".join(expect["key_facts"])}

    monkeypatch.setattr(app.tickets, "summarize_ticket", perfect)
    runpy.run_path(str(RUN_TICKETS), run_name="__main__")
    with open("evals/results.json") as f:
        results = json.load(f)
    assert results["meta"]["model"] == "claude-sonnet-5-5"
    assert results["meta"]["dataset"] == "tickets_v3"
    assert results["summary"]["ALL"] == [1.0, len(expected)]
    assert len(results["records"]) == 3 * len(expected)  # 3 trials each
