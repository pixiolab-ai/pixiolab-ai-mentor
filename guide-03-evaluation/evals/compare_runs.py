"""Chapter 12: compare two harness runs with the ship gate.

Reads two results.json files that run_tickets.py wrote for the same dataset,
turns each into what ship_gate() expects, and prints SHIP or HOLD with its
reasons. Exits with status 1 on HOLD, so it can fail a CI step.

    python evals/compare_runs.py evals/baseline.json evals/results.json
    python evals/compare_runs.py OLD NEW --claim-gain

Not printed in the guide: it is the glue between Chapter 05's harness output
and Chapter 12's gate. cost_per_task comes from each run's meta when you
record it there; without it the cost check compares 0 with 0 and passes.
"""
import json
import math
import sys
from collections import defaultdict

from gate import ship_gate


def per_case_scores(records: list[dict]) -> dict[str, float]:
    """One score per case: the share of its trials that passed."""
    trials = defaultdict(list)
    for r in records:
        trials[r["id"]].append(r["passed"])
    return {case: sum(t) / len(t) for case, t in trials.items()}


def p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else 0


def gate_summary(run: dict, cases: list[str]) -> dict:
    scores = per_case_scores(run["records"])
    return {"scores": [scores[c] for c in cases],  # same case order in both
            "cost_per_task": run["meta"].get("cost_per_task", 0.0),
            "p95_ms": p95([r["ms"] for r in run["records"]])}


def hard_failures(run: dict) -> dict[str, int]:
    """Zero-tolerance counts: here, outputs with an invalid category."""
    return {"invalid_category": sum(
        1 for r in run["records"]
        if r["checks"].get("valid_category") is False)}


def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    with open(paths[0]) as f:
        old = json.load(f)
    with open(paths[1]) as f:
        new = json.load(f)
    cases = sorted(set(per_case_scores(old["records"]))
                   & set(per_case_scores(new["records"])))
    decision, reasons = ship_gate(
        gate_summary(old, cases), gate_summary(new, cases),
        hard_failures(new), claim_gain="--claim-gain" in argv)
    print(f"{decision}: {len(cases)} shared cases "
          f"(one case is {100 / max(len(cases), 1):.1f} points)")
    for reason in reasons:
        print(f"  - {reason}")
    return 0 if decision == "SHIP" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
