import json
import time

class Budget:
    def __init__(self, max_steps=20, max_tokens=200_000,
                 max_seconds=300.0, max_cost_usd=1.0):
        self.limits = {"steps": max_steps, "tokens": max_tokens,
                       "seconds": max_seconds, "cost_usd": max_cost_usd}
        self.used = {"steps": 0, "tokens": 0, "cost_usd": 0.0}
        self.started, self.seen = time.monotonic(), {}

    def charge(self, tokens, cost_usd):       # once per model call
        self.used["steps"] += 1
        self.used["tokens"] += tokens
        self.used["cost_usd"] += cost_usd

    def exceeded(self):                       # name of the limit hit, or None
        used = {**self.used, "seconds": time.monotonic() - self.started}
        return next((k for k, v in used.items() if v >= self.limits[k]), None)

    def repeated(self, tool, args, limit=2):  # same call more than twice?
        key = tool + json.dumps(args, sort_keys=True)
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] > limit
