import functools
import time
from dataclasses import dataclass

import anthropic
import openai
from tenacity import (RetryCallState, Retrying, retry_if_exception,
                      wait_exponential_jitter)

from app.config import FALLBACK_MODEL, PRIMARY_MODEL
from app.llm.metrics import (LLM_FAILURES, LLM_FALLBACKS, LLM_REQUESTS,
                             count_retry)
from app.llm.text import NoText, claude_text

claude = anthropic.Anthropic(max_retries=0)  # our loop is the only retry


@dataclass
class LLMRequest:
    messages: list[dict]
    system: str | None = None
    model: str = PRIMARY_MODEL
    max_tokens: int = 2048   # thinking counts toward this cap
    effort: str = "low"      # raise it per feature where quality needs it
    json_schema: dict | None = None


@dataclass
class LLMResult:
    text: str
    model: str
    fallback: bool
    input_tokens: int        # uncached input only
    output_tokens: int       # includes thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


ATTEMPT_TIMEOUT_S = 20.0  # one call; size it from your p99 latency


def _seconds_left(deadline: float) -> float:
    return min(ATTEMPT_TIMEOUT_S, deadline - time.monotonic())


def _claude_once(req: LLMRequest, deadline: float) -> LLMResult:
    """One attempt, no retries, never past the deadline."""
    config: dict = {"effort": req.effort}
    if req.json_schema:
        config["format"] = {"type": "json_schema", "schema": req.json_schema}
    extra = {"system": req.system} if req.system else {}
    s = _seconds_left(deadline)  # httpx times each phase: cap connect too
    client = claude.with_options(timeout=anthropic.Timeout(
        s, connect=min(s, 3.0)))
    resp = client.messages.create(
        model=req.model, max_tokens=req.max_tokens, messages=req.messages,
        output_config=config, **extra)
    u = resp.usage
    result = LLMResult(
        text="", model=req.model, fallback=False,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0)
    try:
        result.text = claude_text(resp)
    except NoText as exc:  # still billed: attach usage so it gets logged
        raise NoText(exc.reason, result) from None
    return result


RETRY_STATUS = {408, 409, 429}  # plus every 5xx, including 529 overloaded


def provider_failed(exc: BaseException) -> bool:
    """Is the provider, not our request, the problem?"""
    if isinstance(exc, (anthropic.APIConnectionError,
                        openai.APIConnectionError)):
        return True  # dropped connection or timeout
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        return exc.status_code in RETRY_STATUS or exc.status_code >= 500
    return False  # 4xx or NoText: the same call would fail again


def _header(exc: BaseException | None, name: str) -> str | None:
    response = getattr(exc, "response", None)
    return response.headers.get(name) if response is not None else None


def is_retryable(exc: BaseException) -> bool:
    """Worth another attempt on the same provider?"""
    if _header(exc, "x-should-retry") == "false":
        return False  # the server says not to retry; the SDK obeys it too
    return provider_failed(exc)


def retry_after(exc: BaseException | None) -> float | None:
    value = _header(exc, "retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None  # an HTTP date: use our own backoff instead


_backoff = wait_exponential_jitter(initial=1, max=20)  # ~1, 2, 4 s + jitter


def wait_for_retry(rs: RetryCallState) -> float:
    asked = retry_after(rs.outcome.exception())
    return asked if asked is not None else _backoff(rs)


TOTAL_BUDGET_S = 45.0      # retries, waits and the fallback, all of it
FALLBACK_RESERVE_S = 15.0  # held back so the fallback gets a real chance
MIN_ATTEMPT_S = 5.0        # don't start an attempt with less time left


def _with_retries(attempt, req: LLMRequest, deadline: float,
                  max_attempts: int) -> LLMResult:
    def out_of_budget(rs: RetryCallState) -> bool:
        if rs.attempt_number >= max_attempts:
            return True
        # tenacity has already chosen the next sleep: rs.upcoming_sleep
        left = deadline - time.monotonic() - rs.upcoming_sleep
        return left < MIN_ATTEMPT_S

    return Retrying(
        retry=retry_if_exception(is_retryable), wait=wait_for_retry,
        stop=out_of_budget, before_sleep=count_retry,
        reraise=True,  # raise the real error, not tenacity's RetryError
    )(attempt, req, deadline)


def call_claude(req: LLMRequest, deadline: float) -> LLMResult:
    return _with_retries(_claude_once, req, deadline, max_attempts=4)


@functools.cache
def _gpt() -> openai.OpenAI:
    return openai.OpenAI(max_retries=0)  # built on first use: needs a key


def _fallback_once(req: LLMRequest, deadline: float) -> LLMResult:
    """The same request on a second provider, in its API's shape."""
    system = [{"role": "system", "content": req.system}] if req.system else []
    extra = {}
    if req.json_schema:
        extra["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "output", "schema": req.json_schema, "strict": True}}
    s = _seconds_left(deadline)
    client = _gpt().with_options(timeout=openai.Timeout(
        s, connect=min(s, 3.0)))
    resp = client.chat.completions.create(
        model=FALLBACK_MODEL, messages=system + req.messages,
        max_completion_tokens=req.max_tokens, **extra)
    choice, u = resp.choices[0], resp.usage
    details = u.prompt_tokens_details
    cached = (details.cached_tokens or 0) if details else 0
    result = LLMResult(
        text=choice.message.content or "", model=FALLBACK_MODEL,
        fallback=True, input_tokens=u.prompt_tokens - cached,
        output_tokens=u.completion_tokens, cache_read_tokens=cached)
    if choice.finish_reason == "length":
        raise NoText("max_tokens", result)
    if choice.message.refusal or not result.text.strip():
        raise NoText("refusal" if choice.message.refusal else "empty", result)
    return result


class LLMUnavailable(Exception):
    """Every provider failed with errors worth retrying later."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("all model providers failed")
        self.retry_after = retry_after


def generate(messages: list[dict], *, skip_primary: bool = False,
             budget_s: float = TOTAL_BUDGET_S, **options) -> LLMResult:
    """The one entry point every feature uses to call a model."""
    req = LLMRequest(messages=messages, **options)
    deadline = time.monotonic() + budget_s
    LLM_REQUESTS.inc()
    error = None
    if not skip_primary:
        try:
            return call_claude(req, deadline - FALLBACK_RESERVE_S)
        except Exception as exc:
            if not provider_failed(exc):
                raise  # our bug or a refusal: a fallback would hide it
            error = exc
    if not FALLBACK_MODEL or deadline - time.monotonic() < MIN_ATTEMPT_S:
        LLM_FAILURES.inc()
        raise LLMUnavailable(retry_after(error)) from error
    LLM_FALLBACKS.inc()
    try:
        return _with_retries(_fallback_once, req, deadline, max_attempts=2)
    except Exception as exc:
        if not provider_failed(exc):
            raise
        LLM_FAILURES.inc()
        raise LLMUnavailable(retry_after(exc)) from exc
