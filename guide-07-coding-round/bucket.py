import threading
import time

class TokenBucket:
    """Refills `rate` units per second, holds at most `capacity` units."""
    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = rate, capacity
        self.tokens, self.updated = capacity, time.monotonic()
        self.lock = threading.Lock()

    def _take(self, cost: float) -> float:
        """Spend `cost` and return 0, or return the seconds to wait."""
        with self.lock:
            now = time.monotonic()
            elapsed, self.updated = now - self.updated, now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= cost:
                self.tokens -= cost
                return 0.0
            return (cost - self.tokens) / self.rate

    def acquire(self, cost: float = 1.0) -> None:
        if cost > self.capacity:
            raise ValueError("cost exceeds bucket capacity")
        while (wait := self._take(cost)) > 0:
            time.sleep(wait)              # sleep outside the lock
