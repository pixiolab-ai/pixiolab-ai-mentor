"""Chapter 12: the ship gate, and compare_runs.py feeding it two harness runs."""
import json
import random

import compare_runs
from gate import ship_gate


def run_summary(scores, cost=0.010, p95=1200):
    return {"scores": scores, "cost_per_task": cost, "p95_ms": p95}


def baseline_scores(n=400, seed=1):
    rng = random.Random(seed)
    return [float(rng.random() < 0.8) for _ in range(n)]


def test_identical_quality_ships():
    old = baseline_scores()
    decision, reasons = ship_gate(run_summary(old), run_summary(old), {})
    assert decision == "SHIP" and reasons == []


def test_a_hard_failure_blocks_whatever_the_scores_say():
    old = baseline_scores()
    decision, reasons = ship_gate(run_summary(old), run_summary(old),
                                  {"schema_invalid": 0, "safety_regressions": 1})
    assert decision == "HOLD"
    assert reasons == ["hard gate: safety_regressions = 1"]


def test_a_regression_beyond_the_margin_holds():
    old = baseline_scores()
    new = [0.0 if i % 10 == 0 else s for i, s in enumerate(old)]  # ~8% worse
    decision, reasons = ship_gate(run_summary(old), run_summary(new), {})
    assert decision == "HOLD"
    assert reasons[0].startswith("can't rule out a regression")


def test_claimed_gains_must_exclude_zero():
    old = baseline_scores()
    decision, reasons = ship_gate(run_summary(old), run_summary(old), {},
                                  claim_gain=True)
    assert decision == "HOLD"
    assert reasons == ["claimed improvement not shown: CI includes 0"]


def test_cost_and_latency_budgets():
    old = baseline_scores()
    decision, reasons = ship_gate(run_summary(old),
                                  run_summary(old, cost=0.012, p95=1250), {})
    assert decision == "HOLD"
    assert reasons == ["cost_per_task over budget: 0.012 vs 0.01"]


def write_run(path, outcomes, ms=100):
    """A results.json like run_tickets.py writes: 3 trials per case."""
    records = [{"id": case, "trial": t, "error": None, "slice": "billing",
                "ms": ms, "checks": {"valid_category": True},
                "passed": passed}
               for case, trials in outcomes.items()
               for t, passed in enumerate(trials)]
    path.write_text(json.dumps({"meta": {"dataset": "tickets_v3"},
                                "summary": {}, "records": records}))
    return str(path)


def test_compare_runs_ships_an_unchanged_run(tmp_path, capsys):
    outcomes = {f"c{i}": [True, True, i % 4 != 0] for i in range(40)}
    old = write_run(tmp_path / "old.json", outcomes)
    new = write_run(tmp_path / "new.json", outcomes)
    assert compare_runs.main([old, new]) == 0
    assert capsys.readouterr().out.startswith("SHIP: 40 shared cases")


def test_compare_runs_holds_a_broken_run(tmp_path, capsys):
    old_outcomes = {f"c{i}": [True, True, True] for i in range(40)}
    new_outcomes = {c: ([False] * 3 if i < 6 else t)
                    for i, (c, t) in enumerate(old_outcomes.items())}
    old = write_run(tmp_path / "old.json", old_outcomes)
    new = write_run(tmp_path / "new.json", new_outcomes, ms=100)
    assert compare_runs.main([old, new]) == 1
    assert "can't rule out a regression" in capsys.readouterr().out


def test_p95_is_nearest_rank():
    assert compare_runs.p95(list(range(1, 101))) == 95
    assert compare_runs.p95([7]) == 7
