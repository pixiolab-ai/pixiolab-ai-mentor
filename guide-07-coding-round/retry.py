import random
import time

RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}

class APIError(Exception):
    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status, self.retry_after = status, retry_after

def call_with_retry(fn, attempts=5, base=0.5, cap=20.0, deadline_s=30.0):
    """Exponential backoff with full jitter, inside one total deadline."""
    start = time.monotonic()
    for attempt in range(attempts):
        try:
            return fn()
        except APIError as e:
            if e.status not in RETRYABLE or attempt == attempts - 1:
                raise                      # terminal error or out of tries
            delay = random.uniform(0, min(cap, base * 2 ** attempt))
            if e.retry_after is not None:  # the server knows best
                delay = max(delay, e.retry_after)
            if time.monotonic() - start + delay > deadline_s:
                raise                      # waiting would blow the budget
            time.sleep(delay)
