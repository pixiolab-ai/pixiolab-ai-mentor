"""python -m evals.run [--fallback]  ->  evals/results[-fallback].json"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from statistics import mean

import yaml

from app.summarize import summarize_ticket
from evals.scoring import score_case

REPEATS = 3  # majority vote per case: one noisy sample can't flip it
FALLBACK = "--fallback" in sys.argv  # Feature 01: test the backup too


def run_case(case: dict) -> dict:
    run = partial(summarize_ticket, skip_primary=FALLBACK)
    trials = [score_case(case, run) for _ in range(REPEATS)]
    scores = [t["judge"] for t in trials if t["judge"] is not None]
    passes = sum(t["passed"] for t in trials)
    return {"passed": passes * 2 > REPEATS, "passes": passes,
            "judge": mean(scores) if scores else None,
            "critical": case.get("critical", False), "trials": trials}


if __name__ == "__main__":
    with open("evals/golden.yaml") as f:
        cases = yaml.safe_load(f)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip([c["id"] for c in cases],
                           pool.map(run_case, cases)))
    judged = [r["judge"] for r in results.values() if r["judge"] is not None]
    summary = {
        "n_cases": len(results),
        "pass_rate": sum(r["passed"] for r in results.values()) / len(results),
        "avg_judge": mean(judged) if judged else None,
        "unjudged_cases": len(results) - len(judged),
    }
    out = "evals/results-fallback.json" if FALLBACK else "evals/results.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    print(json.dumps(summary, indent=2))
