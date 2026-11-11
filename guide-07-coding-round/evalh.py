from statistics import mean

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def run_eval(cases, system, k=5, repeats=3, gate=0.90):
    """cases: dicts with 'q', 'relevant' (doc ids; empty if out of scope)
    and 'check' (answer -> bool). system(q) -> (retrieved_ids, answer)."""
    rows = []
    for case in cases:
        runs = [system(case["q"]) for _ in range(repeats)]  # outputs vary
        rel = case["relevant"]
        rows.append({
            "q": case["q"],
            "recall": mean(recall_at_k(ids, rel, k) for ids, _ in runs)
                      if rel else None,      # no recall if out of scope
            "pass": mean(case["check"](ans) for _, ans in runs)})
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    pass_rate = mean(r["pass"] for r in rows)
    return {f"recall@{k}": round(mean(recalls), 3) if recalls else None,
            "pass_rate": round(pass_rate, 3),
            "flaky": [r["q"] for r in rows if 0 < r["pass"] < 1],
            "ship": pass_rate >= gate}
