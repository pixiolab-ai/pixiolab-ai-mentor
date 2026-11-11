import os
import time

import redis

from app.config import REDIS_URL

r = redis.Redis.from_url(REDIS_URL)
# Set these to about 80% of your account's limits for the model.
RPM = int(os.getenv("LLM_RPM_LIMIT", "400"))
ITPM = int(os.getenv("LLM_ITPM_LIMIT", "200000"))


def take(name: str, amount: int, per_minute: int) -> float:
    """Reserve `amount` from a budget shared by every worker. Returns 0
    if granted, else the seconds until the next window."""
    window = int(time.time() // 60)
    key = f"ratelimit:{name}:{window}"
    pipe = r.pipeline()            # MULTI/EXEC: both commands or neither
    pipe.incrby(key, amount)
    pipe.expire(key, 120)
    used, _ = pipe.execute()
    if used <= per_minute:
        return 0.0
    r.decrby(key, amount)          # not sending after all: give it back
    return 60 - time.time() % 60


def wait_needed(estimated_input_tokens: int) -> float:
    """0 if the call fits under both limits. (A refused reservation can
    leave the other held until the window ends: conservative.)"""
    return max(take("requests", 1, RPM),
               take("input_tokens", estimated_input_tokens, ITPM))
