"""Chapter 05: route requests, then choose the confidence threshold from data.

1. route() each labeled request with a small, fast model.
2. Record (confidence, was the route right?) for each one.
3. pick_threshold() finds the lowest threshold that meets the target accuracy.
4. handle() sends requests under it to a fallback instead of guessing.

Six requests is far too few to calibrate anything; use a labeled set of a
few hundred from real traffic. Run from guide-02-agents/ (needs
ANTHROPIC_API_KEY; set ROUTER_MODEL to a current small model ID):

    python -m examples.router_demo
"""
from agent.router import handle, pick_threshold, route

LABELED = [
    ("Why does my Python script raise KeyError on line 12?", "coding"),
    ("Summarize recent papers on retrieval evaluation.", "research"),
    ("I was charged twice for order ORD-20481234.", "support"),
    ("My refund for ORD-55510001 hasn't arrived.", "support"),
    ("Write a SQL query that counts signups per day.", "coding"),
    ("hello?", "unclear"),
]

SPECIALISTS = {
    "coding": lambda request, r: f"coding specialist takes: {request}",
    "research": lambda request, r: f"research specialist takes: {request}",
    "support": lambda request, r: f"support specialist takes order "
                                  f"{r.order_id}: {request}",
}


def clarify(request: str) -> str:
    return f"clarifying question for: {request}"


def main() -> float:
    scored = []
    for request, label in LABELED:
        r = route(request)
        scored.append((r.confidence, r.category == label))
        print(f"{r.category:>8} {r.confidence:.2f} {r.order_id or '-':>12}  "
              f"{'ok ' if r.category == label else 'MISS'} {request}")
    threshold = pick_threshold(scored, target=0.95)
    print(f"\nthreshold for 95% routed accuracy: {threshold}")
    print(handle("My refund for ORD-55510001 hasn't arrived.", SPECIALISTS,
                 clarify, threshold))
    return threshold


if __name__ == "__main__":
    main()
