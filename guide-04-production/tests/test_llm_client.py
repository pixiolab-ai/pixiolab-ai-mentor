"""Chapter 00 and Feature 01: reading output safely, retries, time budget, fallback.

No test calls a real provider. Most replace the SDK clients with small fakes
that return scripted responses or raise scripted errors; the last two run
the real SDKs against an in-memory HTTP transport, to check the request
bodies the providers would receive.
"""
import json
import time
from types import SimpleNamespace as NS

import anthropic
import httpx2
import openai
import pytest
from prometheus_client import REGISTRY

import app.llm.client as client_mod
from app.llm.text import NoText

URL = "https://api.anthropic.com/v1/messages"
USER = [{"role": "user", "content": "Say hi"}]


# ---------- fakes ----------

def status_error(cls, code: int, headers: dict | None = None):
    """An SDK error carrying an HTTP status and headers, as the SDK raises it."""
    response = httpx2.Response(code, request=httpx2.Request("POST", URL),
                               headers=headers or {})
    return cls("simulated error", response=response, body=None)


def claude_message(text="hi", stop="end_turn", thinking=True):
    """An Anthropic Message: optionally a thinking block first, then text."""
    blocks = [NS(type="thinking", thinking="", signature="sig")] if thinking else []
    if text is not None:
        blocks.append(NS(type="text", text=text))
    usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=None,
               cache_creation_input_tokens=3)
    return NS(content=blocks, stop_reason=stop, usage=usage)


def gpt_completion(text="fallback hi", finish="stop", refusal=None):
    message = NS(content=text, refusal=refusal)
    usage = NS(prompt_tokens=20, completion_tokens=7,
               prompt_tokens_details=NS(cached_tokens=4))
    return NS(choices=[NS(message=message, finish_reason=finish)], usage=usage)


class Scripted:
    """Stands in for client.messages or client.chat.completions."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClaude:
    def __init__(self, script):
        self.messages = Scripted(script)
        self.timeouts = []

    def with_options(self, timeout):
        self.timeouts.append(timeout)
        return self


class FakeGPT:
    def __init__(self, script):
        self.completions = Scripted(script)
        self.chat = NS(completions=self.completions)
        self.timeouts = []

    def with_options(self, timeout):
        self.timeouts.append(timeout)
        return self


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Make waits instant, but record them. Backoff without a server hint is
    fixed at 1.5 s so the tests are deterministic."""
    waited = []
    monkeypatch.setattr(client_mod, "_backoff", lambda retry_state: 1.5)
    monkeypatch.setattr(time, "sleep", waited.append)
    return waited


def install(monkeypatch, claude_script, gpt_script=(), fallback="gpt-test"):
    claude, gpt = FakeClaude(claude_script), FakeGPT(gpt_script)
    monkeypatch.setattr(client_mod, "claude", claude)
    monkeypatch.setattr(client_mod, "_gpt", lambda: gpt)
    monkeypatch.setattr(client_mod, "FALLBACK_MODEL", fallback)
    return claude, gpt


# ---------- Chapter 00: read output by block type ----------

def test_thinking_block_first_is_skipped(monkeypatch):
    claude, _ = install(monkeypatch, [claude_message("hello")])
    r = client_mod.generate(USER, system="be brief")
    assert r.text == "hello" and not r.fallback
    assert (r.input_tokens, r.output_tokens) == (10, 5)
    assert (r.cache_read_tokens, r.cache_write_tokens) == (0, 3)
    sent = claude.messages.calls[0]
    assert sent["system"] == "be brief"
    assert sent["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("stop, text, reason", [
    ("refusal", None, "refusal"),
    ("max_tokens", "a partial answ", "max_tokens"),
    ("end_turn", None, "empty"),
])
def test_no_usable_text_raises_notext(monkeypatch, stop, text, reason):
    claude, gpt = install(monkeypatch, [claude_message(text, stop=stop)],
                          [gpt_completion()])
    with pytest.raises(NoText) as caught:
        client_mod.generate(USER)
    assert caught.value.reason == reason
    assert caught.value.result.input_tokens == 10  # billed, so still logged
    assert len(claude.messages.calls) == 1         # never retried
    assert not gpt.completions.calls               # never hidden by fallback


# ---------- Feature 01, Step 2: retry only what can recover ----------

def test_overloaded_529_is_retried(monkeypatch):
    overloaded = status_error(anthropic.InternalServerError, 529)
    claude, _ = install(monkeypatch, [overloaded, overloaded,
                                      claude_message("ok")])
    assert client_mod.generate(USER).text == "ok"
    assert len(claude.messages.calls) == 3


def test_bad_request_is_not_retried_and_not_hidden(monkeypatch):
    claude, gpt = install(monkeypatch,
                          [status_error(anthropic.BadRequestError, 400)])
    with pytest.raises(anthropic.BadRequestError):
        client_mod.generate(USER)
    assert len(claude.messages.calls) == 1 and not gpt.completions.calls


def test_retry_after_is_honored_exactly(monkeypatch, sleeps):
    limited = status_error(anthropic.RateLimitError, 429, {"retry-after": "2"})
    install(monkeypatch, [limited, claude_message("ok")])
    client_mod.generate(USER)
    assert sleeps == [2.0]


def test_retry_after_longer_than_budget_goes_to_fallback(monkeypatch, sleeps):
    limited = status_error(anthropic.RateLimitError, 429, {"retry-after": "60"})
    _, gpt = install(monkeypatch, [limited], [gpt_completion()])
    r = client_mod.generate(USER, system="be brief")
    assert r.fallback and r.text == "fallback hi"
    assert sleeps == []  # didn't sleep through the user's deadline
    assert r.input_tokens == 16 and r.cache_read_tokens == 4  # 20 - 4 cached
    first = gpt.completions.calls[0]["messages"][0]
    assert first == {"role": "system", "content": "be brief"}


def test_should_retry_false_skips_retries(monkeypatch):
    capped = status_error(anthropic.RateLimitError, 429,
                          {"x-should-retry": "false"})
    claude, _ = install(monkeypatch, [capped], [gpt_completion()])
    assert client_mod.generate(USER).fallback
    assert len(claude.messages.calls) == 1


def test_exhausted_retries_fall_back(monkeypatch):
    error = status_error(anthropic.InternalServerError, 500)
    claude, _ = install(monkeypatch, [error] * 4, [gpt_completion()])
    assert client_mod.generate(USER).fallback
    assert len(claude.messages.calls) == 4  # max_attempts for the primary


def test_no_fallback_raises_unavailable_with_server_wait(monkeypatch):
    error = status_error(anthropic.InternalServerError, 503,
                         {"retry-after": "7"})
    install(monkeypatch, [error] * 4, fallback=None)
    with pytest.raises(client_mod.LLMUnavailable) as caught:
        client_mod.generate(USER)
    assert caught.value.retry_after == 7.0  # Feature 05's queue uses it


def test_fallback_bug_is_raised_not_swallowed(monkeypatch):
    primary_down = status_error(anthropic.InternalServerError, 500,
                                {"x-should-retry": "false"})
    install(monkeypatch, [primary_down],
            [status_error(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        client_mod.generate(USER)


def test_skip_primary_goes_straight_to_fallback(monkeypatch):
    claude, _ = install(monkeypatch, [], [gpt_completion()])
    assert client_mod.generate(USER, skip_primary=True).fallback
    assert not claude.messages.calls


def test_json_schema_reaches_both_providers(monkeypatch):
    schema = {"type": "object"}
    primary_down = status_error(anthropic.InternalServerError, 500,
                                {"x-should-retry": "false"})
    claude, gpt = install(monkeypatch, [primary_down],
                          [gpt_completion('{"a": 1}')])
    client_mod.generate(USER, json_schema=schema)
    assert claude.messages.calls[0]["output_config"]["format"]["schema"] is schema
    response_format = gpt.completions.calls[0]["response_format"]
    assert response_format["json_schema"]["strict"] is True


# ---------- Feature 01, Step 3: one total time budget ----------

def test_total_budget_holds_when_every_attempt_times_out(monkeypatch):
    """Every attempt hangs until its timeout. A fake clock advances by each
    attempt's timeout and each sleep: the whole call must end within 45 s,
    with the fallback still getting a turn."""
    now = [1000.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    claude, gpt = FakeClaude([]), FakeGPT([])

    def hang(fake, error_cls):
        def create(**kwargs):
            fake_calls.append(fake)
            now[0] += fake.timeouts[-1].read  # the attempt uses its timeout
            raise error_cls(request=httpx2.Request("POST", URL))
        return create

    fake_calls = []
    claude.messages.create = hang(claude, anthropic.APITimeoutError)
    gpt.completions.create = hang(gpt, openai.APITimeoutError)
    monkeypatch.setattr(client_mod, "claude", claude)
    monkeypatch.setattr(client_mod, "_gpt", lambda: gpt)
    monkeypatch.setattr(client_mod, "FALLBACK_MODEL", "gpt-test")

    start = now[0]
    with pytest.raises(client_mod.LLMUnavailable):
        client_mod.generate(USER)
    assert now[0] - start <= client_mod.TOTAL_BUDGET_S + 1e-9
    assert gpt in fake_calls  # the reserve gave the fallback a real chance
    assert all(t.connect <= 3.0 for t in claude.timeouts + gpt.timeouts)


# ---------- Feature 01, Step 5: counters ----------

def test_retries_are_counted_by_error_type(monkeypatch):
    labels = {"error": "InternalServerError"}
    before = REGISTRY.get_sample_value("llm_retries_total", labels) or 0
    error = status_error(anthropic.InternalServerError, 500)
    install(monkeypatch, [error, claude_message()])
    client_mod.generate(USER)
    assert REGISTRY.get_sample_value("llm_retries_total", labels) == before + 1


# ---------- the real SDKs, on an in-memory transport ----------

def test_anthropic_request_body(monkeypatch):
    import app.summarize as summarize_mod
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                        {"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 7,
                      "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}})

    sdk = anthropic.Anthropic(
        api_key="test", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(client_mod, "claude", sdk)
    r = client_mod.generate(USER, system="sys",
                            json_schema=summarize_mod.SCHEMA)
    body = bodies[0]
    assert r.text == "hello" and body["system"] == "sys"
    assert body["output_config"]["format"]["schema"] == summarize_mod.SCHEMA
    assert "temperature" not in body  # sampling settings are never sent


def test_openai_request_body(monkeypatch):
    import app.summarize as summarize_mod
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "refusal": None,
                "content": '{"category": "account", "urgency": "low", '
                           '"summary": "x"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                      "total_tokens": 13,
                      "prompt_tokens_details": {"cached_tokens": 2}}})

    sdk = openai.OpenAI(
        api_key="test", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(client_mod, "_gpt", lambda: sdk)
    monkeypatch.setattr(client_mod, "FALLBACK_MODEL", "gpt-test")
    r = client_mod.generate(USER, system="sys", skip_primary=True,
                            json_schema=summarize_mod.SCHEMA)
    body = bodies[0]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert r.fallback and r.input_tokens == 8 and r.cache_read_tokens == 2
