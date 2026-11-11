"""Chapter 06: the binary faithfulness judge, and the feature under test.

The judge's client is either a fake that returns a scripted response or the
real SDK on an in-memory transport, so these tests check the request it
sends and how it treats refusals and truncation, without a real model.
"""
import json
from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

import app.tickets as tickets
import judges.faithfulness as faithfulness


def message(text=None, stop="end_turn"):
    blocks = [NS(type="thinking", thinking="", signature="s")]
    if text is not None:
        blocks.append(NS(type="text", text=text))
    return NS(content=blocks, stop_reason=stop)


def fake_client(resp):
    return NS(messages=NS(create=lambda **request: resp))


VERDICT = {"claims": [{"claim": "Returns are accepted for 60 days.",
                       "evidence": "within 60 days", "supported": True}],
           "verdict": "PASS"}


def test_a_pass_comes_back_with_its_claims_and_evidence(monkeypatch):
    monkeypatch.setattr(faithfulness, "client",
                        fake_client(message(json.dumps(VERDICT))))
    out = faithfulness.judge_faithfulness("Return within 60 days.",
                                          "Returns are accepted for 60 days.")
    assert out == VERDICT


@pytest.mark.parametrize("resp", [
    message(None, stop="refusal"),                 # declined
    message('{"claims": [', stop="max_tokens"),    # cut off mid-JSON
    message(None, stop="end_turn"),                # no text at all
])
def test_no_usable_answer_is_an_error_never_a_pass_or_fail(monkeypatch, resp):
    monkeypatch.setattr(faithfulness, "client", fake_client(resp))
    out = faithfulness.judge_faithfulness("ctx", "answer")
    assert out["verdict"] == "ERROR"


def test_the_request_pins_the_rubric_and_the_schema(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": faithfulness.JUDGE_MODEL, "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": json.dumps(VERDICT)}],
            "usage": {"input_tokens": 50, "output_tokens": 40}})

    sdk = anthropic.Anthropic(
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(faithfulness, "client", sdk)
    faithfulness.judge_faithfulness("the context", "the answer")
    body = bodies[0]
    assert body["model"] == faithfulness.JUDGE_MODEL
    assert body["system"] == faithfulness.RUBRIC
    assert body["output_config"]["format"]["schema"] == faithfulness.SCHEMA
    assert "<context>\nthe context\n</context>" in body["messages"][0]["content"]


def test_ticket_summarizer_returns_the_schema_fields(monkeypatch):
    requests = []

    def create(**request):
        requests.append(request)
        return message('{"category": "billing", "summary": "Charged twice."}')

    monkeypatch.setattr(tickets, "client", NS(messages=NS(create=create)))
    assert tickets.summarize_ticket("I was charged twice") == {
        "category": "billing", "summary": "Charged twice."}
    fmt = requests[0]["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": tickets.SCHEMA}


def test_ticket_summarizer_raises_on_a_refusal(monkeypatch):
    monkeypatch.setattr(tickets, "client",
                        fake_client(message(None, stop="refusal")))
    with pytest.raises(RuntimeError, match="refusal"):
        tickets.summarize_ticket("...")
