import os
from typing import Literal

import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-5-5")  # small, fast

class Route(BaseModel):
    category: Literal["coding", "research", "support", "unclear"]
    confidence: float          # 0 to 1, self-reported, so calibrate it
    order_id: str | None = None

SYSTEM = ("Classify the user's request. Use 'unclear' if it fits no "
          "category or spans several. Extract an order ID if present.")

def route(request: str) -> Route:
    return client.messages.parse(             # schema-checked output
        model=ROUTER_MODEL, max_tokens=2048, system=SYSTEM,
        messages=[{"role": "user", "content": request}],
        output_format=Route).parsed_output

def handle(request, specialists, fallback, threshold):
    r = route(request)
    if r.category == "unclear" or r.confidence < threshold:
        return fallback(request)              # clarify or general handler
    return specialists[r.category](request, r)

def pick_threshold(scored, target=0.95):
    """Lowest threshold whose routed accuracy on a labeled set of
    (confidence, was_correct) pairs meets the target."""
    for t in (i / 20 for i in range(21)):
        kept = [ok for conf, ok in scored if conf >= t]
        if kept and sum(kept) / len(kept) >= target:
            return t
    return 1.01                               # never route automatically
