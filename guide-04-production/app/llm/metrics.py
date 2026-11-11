from prometheus_client import Counter

LLM_REQUESTS = Counter("llm_requests_total", "generate() calls")
LLM_RETRIES = Counter("llm_retries_total", "Retried attempts", ["error"])
LLM_FALLBACKS = Counter("llm_fallbacks_total", "Sent to the fallback")
LLM_FAILURES = Counter("llm_failures_total", "Failed on every provider")


def count_retry(retry_state) -> None:
    """tenacity before_sleep hook: runs once before each retry."""
    exc = retry_state.outcome.exception()
    LLM_RETRIES.labels(error=type(exc).__name__).inc()
