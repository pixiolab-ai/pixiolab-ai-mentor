"""9.06 streamed tool calls, 9.07 the eval harness, 9.08 bounded fan-out."""
import asyncio
import itertools
from types import SimpleNamespace as NS

import pytest

import fanout
from evalh import recall_at_k, run_eval
from toolstream import collect_tool_calls


# ---------- 9.06 reassemble streamed tool calls ----------

def tool_stream(*calls, stop="tool_use"):
    """Anthropic stream events for a text block, then each tool call with its
    arguments split into JSON fragments."""
    events = [{"type": "message_start", "message": {}},
              {"type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}},
              {"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "Checking"}},
              {"type": "content_block_stop", "index": 0}]
    for i, (call_id, name, fragments) in enumerate(calls, start=1):
        events.append({"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": call_id,
                                         "name": name, "input": {}}})
        events += [{"type": "content_block_delta", "index": i,
                    "delta": {"type": "input_json_delta",
                              "partial_json": part}} for part in fragments]
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop},
                   "usage": {}})
    return events + [{"type": "message_stop"}]


def test_fragments_split_mid_string_are_joined_before_parsing():
    events = tool_stream(
        ("toolu_1", "get_weather", ["", '{"city": "Par', 'is", "unit": "c"}']),
        ("toolu_2", "now", []))
    assert collect_tool_calls(events) == [
        {"id": "toolu_1", "name": "get_weather",
         "input": {"city": "Paris", "unit": "c"}},
        {"id": "toolu_2", "name": "now", "input": {}}]  # no args: {}


def test_a_stream_that_ends_inside_a_call_raises():
    events = tool_stream(("toolu_1", "get_weather", ['{"city": "Par']))
    close = next(i for i, e in enumerate(events)
                 if e["type"] == "content_block_stop" and e["index"] == 1)
    with pytest.raises(ValueError, match="inside a tool call"):
        collect_tool_calls(events[:close])  # the connection dropped here


def test_max_tokens_truncation_raises():
    events = tool_stream(("toolu_1", "search", ['{"q": "x"}']),
                         stop="max_tokens")
    with pytest.raises(ValueError, match="truncated"):
        collect_tool_calls(events)


# ---------- 9.07 eval harness with a ship gate ----------

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 2) == 0.5


def test_repeats_expose_flaky_cases_and_gate_on_pass_rate():
    flip = itertools.cycle([True, False, True])

    def system(question):
        if question == "q2":
            return ["x"], "good" if next(flip) else "bad"
        return ["d1", "d2"], "good"

    good = lambda answer: answer == "good"
    cases = [{"q": "q1", "relevant": {"d1"}, "check": good},
             {"q": "q2", "relevant": {"d1"}, "check": good},
             {"q": "q3", "relevant": set(), "check": good}]  # out of scope
    report = run_eval(cases, system, k=5, repeats=3)
    assert report["recall@5"] == 0.5          # q3 has no recall to measure
    assert report["flaky"] == ["q2"]          # passed 2 of 3 runs
    assert report["pass_rate"] == round((1 + 2 / 3 + 1) / 3, 3)  # 0.889
    assert report["ship"] is False            # below the default 0.90 gate


# ---------- 9.08 bounded concurrent model calls ----------

def test_at_most_limit_calls_in_flight_and_results_keep_input_order():
    in_flight = {"now": 0, "max": 0}

    async def work(x):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        if x == 3:
            raise RuntimeError("boom")
        return x * 2

    results = asyncio.run(fanout.map_bounded(work, list(range(16)), limit=4))
    assert in_flight["max"] == 4
    assert results[0] == 0 and results[15] == 30
    assert isinstance(results[3], RuntimeError)  # one failure, in its slot


def test_a_slow_call_times_out_without_canceling_the_batch():
    async def work(x):
        await asyncio.sleep(1 if x == "slow" else 0)
        return x

    results = asyncio.run(fanout.map_bounded(work, ["fast", "slow"],
                                             timeout_s=0.05))
    assert results[0] == "fast"
    assert isinstance(results[1], (asyncio.TimeoutError, TimeoutError))


def test_summarize_joins_text_blocks_only(monkeypatch):
    requests = []

    async def create(**request):
        requests.append(request)
        return NS(content=[NS(type="thinking", thinking=""),
                           NS(type="text", text="A short summary.")])

    monkeypatch.setattr(fanout, "client", NS(messages=NS(create=create)))
    assert asyncio.run(fanout.summarize("a long document")) == "A short summary."
    assert requests[0]["model"] == fanout.MODEL
