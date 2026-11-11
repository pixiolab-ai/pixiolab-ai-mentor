"""Chapter 05: routing with a threshold chosen from labeled data."""
import json
from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

import agent.router as router


def test_threshold_is_the_lowest_that_meets_the_target():
    scored = ([(0.2, False)] * 10 + [(0.6, True)] * 10
              + [(0.95, True)] * 10)
    assert router.pick_threshold(scored) == pytest.approx(0.25)


def test_threshold_zero_when_every_route_is_accurate_enough():
    assert router.pick_threshold([(0.9, True)] * 19 + [(0.9, False)]) == 0.0


def test_never_route_automatically_when_no_threshold_works():
    assert router.pick_threshold([(0.5, False)]) == 1.01


@pytest.mark.parametrize("category, confidence, expected", [
    ("support", 0.8, "specialist"),
    ("support", 0.5, "fallback"),   # below the threshold: don't guess
    ("unclear", 0.99, "fallback"),
])
def test_handle_sends_low_confidence_to_the_fallback(
        monkeypatch, category, confidence, expected):
    monkeypatch.setattr(router, "route", lambda request: router.Route(
        category=category, confidence=confidence))
    specialists = {"support": lambda request, r: "specialist"}
    assert router.handle("x", specialists, lambda r: "fallback", 0.7) == expected


def test_route_asks_for_schema_checked_output(monkeypatch):
    """The real SDK's messages.parse, on an in-memory transport."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": router.ROUTER_MODEL, "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": json.dumps(
                {"category": "support", "confidence": 0.92,
                 "order_id": "ORD-20481234"})}],
            "usage": {"input_tokens": 20, "output_tokens": 15}})

    sdk = anthropic.Anthropic(
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(router, "client", sdk)
    r = router.route("I was charged twice for order ORD-20481234.")
    assert r == router.Route(category="support", confidence=0.92,
                             order_id="ORD-20481234")
    body = bodies[0]
    assert body["model"] == router.ROUTER_MODEL
    assert body["system"] == router.SYSTEM
    schema = body["output_config"]["format"]["schema"]
    assert set(schema["properties"]["category"]["enum"]) == {
        "coding", "research", "support", "unclear"}


def test_route_returns_parsed_output(monkeypatch):
    parsed = router.Route(category="coding", confidence=0.7)
    monkeypatch.setattr(router, "client", NS(messages=NS(
        parse=lambda **kw: NS(parsed_output=parsed))))
    assert router.route("fix my bug") is parsed
