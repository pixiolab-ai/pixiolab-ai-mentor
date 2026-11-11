import time
from collections import defaultdict

def grade(graders, case, output):
    results = {}
    for name, check in graders.items():
        try:
            results[name] = bool(check(case, output))
        except Exception:              # malformed output fails the check
            results[name] = False
    return results

def run_suite(cases, system, graders, trials=3):
    records = []
    for case in cases:
        for trial in range(trials):
            t0 = time.perf_counter()
            try:
                out, err = system(case["input"]), None
            except Exception as exc:   # a crash is a recorded failure
                out, err = None, repr(exc)
            checks = {} if err else grade(graders, case, out)
            records.append({
                "id": case["id"], "trial": trial, "error": err,
                "slice": case.get("slice", {}).get("intent", "none"),
                "ms": round((time.perf_counter() - t0) * 1000),
                "checks": checks,
                "passed": bool(checks) and all(checks.values())})
    return records

def summarize(records):  # pass rate per slice, trials averaged per case
    per_case = defaultdict(list)
    for r in records:
        per_case[(r["slice"], r["id"])].append(r["passed"])
    by_slice = defaultdict(list)
    for (slice_name, _), runs in per_case.items():
        for key in (slice_name, "ALL"):
            by_slice[key].append(sum(runs) / len(runs))
    return {k: (round(sum(v) / len(v), 3), len(v))
            for k, v in by_slice.items()}