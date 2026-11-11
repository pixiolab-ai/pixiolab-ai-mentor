"""Feature 03: the summarizer, the scorer, the judge call and the regression gate.

The summarizer's model call and the judge are replaced with fakes, so this
checks the suite's logic. To score real outputs, run `python -m evals.run`.
"""
import importlib.util
import json
import runpy
import shutil
from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest
import yaml

import app.summarize as summarize_mod
import evals.run as run_mod
import evals.scoring as scoring_mod
from app.llm.client import LLMResult
from app.summarize import TicketSummary

CASE = {"input": "I was charged twice", "key_facts": ["charged twice"],
        "expect": {"category": "billing", "urgency": "high"}}


def model_says(text: str):
    def fake(user_id, feature, messages, **options):
        fake.options = options
        return LLMResult(text, "claude-sonnet-5-5", False, 1, 1)
    return fake


def test_summarizer_sends_the_versioned_prompt_settings(monkeypatch):
    fake = model_says('{"category": "billing", "urgency": "high", '
                      '"summary": "Charged twice."}')
    monkeypatch.setattr(summarize_mod, "tracked_generate", fake)
    out = summarize_mod.summarize_ticket("I was charged twice")
    assert out == TicketSummary(category="billing", urgency="high",
                                summary="Charged twice.")
    prompt = yaml.safe_load(open("prompts/ticket_summary.yaml"))
    assert fake.options["model"] == prompt["model"]
    assert fake.options["system"] == prompt["system"]
    assert fake.options["json_schema"] is summarize_mod.SCHEMA


def test_summarizer_rejects_an_answer_outside_the_schema(monkeypatch):
    monkeypatch.setattr(summarize_mod, "tracked_generate", model_says(
        '{"category": "refunds", "urgency": "high", "summary": "x"}'))
    with pytest.raises(ValueError):  # pydantic.ValidationError
        summarize_mod.summarize_ticket("x")


def test_score_case_checks_code_first_then_the_judge(monkeypatch):
    monkeypatch.setattr(scoring_mod, "judge", lambda t, s, f: scoring_mod.Verdict(
        score=4, reason="keeps the facts"))
    good = lambda text: TicketSummary(category="billing", urgency="high",
                                      summary="Charged twice.")
    trial = scoring_mod.score_case(CASE, good)
    assert trial["passed"] and trial["judge"] == 4
    assert trial["checks"] == {"category": True, "urgency": True}


def test_a_crash_is_a_recorded_failure_not_an_aborted_run():
    def broken(text):
        raise ValueError("bad json")
    assert scoring_mod.score_case(CASE, broken) == {
        "passed": False, "judge": None, "error": "ValueError('bad json')"}


def test_a_judge_failure_is_none_not_zero(monkeypatch):
    def judge_down(*args):
        raise RuntimeError("judge down")
    monkeypatch.setattr(scoring_mod, "judge", judge_down)
    good = lambda text: TicketSummary(category="billing", urgency="high",
                                      summary="s")
    trial = scoring_mod.score_case(CASE, good)
    assert trial["passed"] and trial["judge"] is None
    assert "judge down" in trial["judge_error"]


def test_judge_uses_structured_output_and_a_different_model(monkeypatch):
    """Runs the real SDK's messages.parse against an in-memory transport."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": scoring_mod.JUDGE_MODEL, "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                        {"type": "text",
                         "text": '{"score": 3, "reason": "misses one fact"}'}],
            "usage": {"input_tokens": 5, "output_tokens": 5}})

    sdk = anthropic.Anthropic(
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(scoring_mod, "judge_client", sdk)
    verdict = scoring_mod.judge("ticket", "summary", ["a", "b"])
    assert verdict == scoring_mod.Verdict(score=3, reason="misses one fact")
    body = bodies[0]
    assert body["model"] != summarize_mod.load_prompt("ticket_summary")["model"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "<key_facts>a; b</key_facts>" in body["messages"][0]["content"]


def test_judge_without_a_verdict_raises(monkeypatch):
    refusal = NS(parsed_output=None, stop_reason="refusal")
    monkeypatch.setattr(scoring_mod, "judge_client",
                        NS(messages=NS(parse=lambda **kw: refusal)))
    with pytest.raises(ValueError, match="refusal"):
        scoring_mod.judge("t", "s", ["a"])


def test_majority_vote_over_three_repeats(monkeypatch):
    outcomes = iter([True, False, True])
    monkeypatch.setattr(run_mod, "score_case", lambda case, run: {
        "passed": next(outcomes), "judge": 4})
    result = run_mod.run_case({"id": "x", "input": "x"})
    assert result["passed"] and result["passes"] == 2


def load_regression_test(path):
    spec = importlib.util.spec_from_file_location("regression_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_no_regression


def test_suite_writes_results_and_the_gate_blocks_a_critical_failure(
        tmp_path, monkeypatch):
    """The whole Feature 03 loop in a temporary folder: run the suite, save a
    baseline, then break cases and watch the gate's verdict."""
    shutil.copytree("evals", tmp_path / "evals",
                    ignore=shutil.ignore_patterns("*.json", "__pycache__"))
    shutil.copytree("tests", tmp_path / "tests",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.chdir(tmp_path)

    cases = yaml.safe_load(open("evals/golden.yaml"))
    right = {c["input"]: c["expect"] for c in cases}

    def perfect_summarizer(ticket, skip_primary=False):
        return TicketSummary(**right[ticket], summary="s")

    monkeypatch.setattr(summarize_mod, "summarize_ticket", perfect_summarizer)
    monkeypatch.setattr(scoring_mod, "judge",
                        lambda t, s, f: scoring_mod.Verdict(score=4, reason="ok"))
    runpy.run_path("evals/run.py", run_name="__main__")
    shutil.copy("evals/results.json", "evals/baseline.json")
    gate = load_regression_test(tmp_path / "tests" / "test_regression.py")
    gate()  # unchanged: passes

    data = json.load(open("evals/results.json"))
    data["cases"]["password-reset"]["passed"] = False  # one net regression
    json.dump(data, open("evals/results.json", "w"))
    gate()  # within the tolerance of one case

    data["cases"]["double-charge"]["passed"] = False   # a critical case
    json.dump(data, open("evals/results.json", "w"))
    with pytest.raises(AssertionError, match="critical cases failed"):
        gate()
