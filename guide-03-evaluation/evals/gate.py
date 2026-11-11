from stats import paired_bootstrap   # from Chapter 4

def ship_gate(old, new, hard_failures, margin=0.02, budget=1.10,
              claim_gain=False):
    """old, new: summaries of two runs on the same cases. Each holds
    "scores" (one per case, mean over trials, same case order),
    "cost_per_task" and "p95_ms". hard_failures: {check: count}."""
    reasons = [f"hard gate: {name} = {count}"
               for name, count in hard_failures.items() if count]
    diff, lo, hi = paired_bootstrap(old["scores"], new["scores"])
    if lo <= -margin:
        reasons.append(f"can't rule out a regression: {diff:+.3f} "
                       f"[{lo:+.3f}, {hi:+.3f}], margin {margin}")
    if claim_gain and lo <= 0:
        reasons.append("claimed improvement not shown: CI includes 0")
    for key in ("cost_per_task", "p95_ms"):
        if new[key] > budget * old[key]:
            reasons.append(f"{key} over budget: {new[key]} vs {old[key]}")
    return ("SHIP" if not reasons else "HOLD"), reasons