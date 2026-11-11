"""9.01 retry with backoff, jitter and a deadline; 9.02 the token bucket.

Both run on a fake clock: sleeping advances it instantly, so the tests are
exact and fast instead of timing-dependent.
"""
import pytest

import bucket
import retry


class FakeClock:
    """Like a real clock, a sleep always lets at least a microsecond pass."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += max(seconds, 1e-6)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    for module in (retry, bucket):
        monkeypatch.setattr(module.time, "monotonic", fake.monotonic)
        monkeypatch.setattr(module.time, "sleep", fake.sleep)
    return fake


def failing(*statuses, then="ok"):
    """A callable that raises the given HTTP statuses, then succeeds."""
    queue = list(statuses)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if queue:
            status = queue.pop(0)
            raise status if isinstance(status, retry.APIError) else \
                retry.APIError(status)
        return then
    fn.calls = calls
    return fn


# ---------- 9.01 retry ----------

def test_transient_errors_are_retried_until_success(clock):
    fn = failing(503, 529)
    assert retry.call_with_retry(fn) == "ok" and fn.calls["n"] == 3


def test_a_400_fails_fast(clock):
    fn = failing(400)
    with pytest.raises(retry.APIError) as caught:
        retry.call_with_retry(fn)
    assert caught.value.status == 400 and fn.calls["n"] == 1
    assert clock.sleeps == []


def test_full_jitter_stays_under_the_exponential_cap(clock):
    retry.call_with_retry(failing(503, 503, 503, 503), attempts=5)
    caps = [0.5, 1.0, 2.0, 4.0]  # base * 2 ** attempt
    assert len(clock.sleeps) == 4
    assert all(0 <= s <= cap for s, cap in zip(clock.sleeps, caps))


def test_retry_after_raises_the_wait(clock):
    retry.call_with_retry(failing(retry.APIError(429, retry_after=3.0)))
    assert clock.sleeps == [3.0]


def test_a_wait_that_would_blow_the_deadline_raises_instead(clock):
    fn = failing(retry.APIError(429, retry_after=100))
    with pytest.raises(retry.APIError):
        retry.call_with_retry(fn, deadline_s=30)
    assert clock.sleeps == []


def test_out_of_attempts_raises_the_last_error(clock):
    fn = failing(503, 503, 503, 503)
    with pytest.raises(retry.APIError):
        retry.call_with_retry(fn, attempts=4)
    assert fn.calls["n"] == 4


# ---------- 9.02 token bucket ----------

def test_a_full_bucket_bursts_then_paces_at_the_refill_rate(clock):
    b = bucket.TokenBucket(rate=100, capacity=10)
    start = clock.now
    for _ in range(30):
        b.acquire(1)
    # 10 units leave at once, the other 20 at 100 per second.
    assert clock.now - start == pytest.approx(0.2, abs=1e-3)


def test_the_wait_is_computed_not_polled(clock):
    b = bucket.TokenBucket(rate=2, capacity=4)
    b.acquire(4)
    b.acquire(3)        # needs 3 more units at 2 per second
    assert clock.sleeps[0] == pytest.approx(1.5)
    assert sum(clock.sleeps) == pytest.approx(1.5)


def test_a_cost_above_capacity_can_never_succeed(clock):
    with pytest.raises(ValueError):
        bucket.TokenBucket(rate=100, capacity=10).acquire(11)
