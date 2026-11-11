"""Chapters 03, 04 and 06: planning, reflection and multi-agent skeletons.

These skeletons take their model calls as plain functions, so the tests pass
deterministic stand-ins and check the control flow the guide describes.
"""
import asyncio

from agent.planning import Plan, Step, run_plan
from agent.reflection import Verdict, reflect
from agent.team import Delegation, SupervisorDecision, run_team


# ---------- Chapter 03: planning ----------

def four_step_plan(goal):
    return Plan(goal=goal, steps=[
        Step(id=1, description="a"), Step(id=2, description="b"),
        Step(id=3, description="c", depends_on=[1, 2]),
        Step(id=4, description="d", depends_on=[3])])


def test_a_revised_plan_runs_immediately_and_keeps_finished_work():
    ran = []

    def executor(step, inputs):
        ran.append(step.id)
        if step.id == 3:
            return {"ok": False, "error": "search failed"}
        return {"ok": True, "result": f"r{step.id}<-{sorted(inputs)}"}

    def replanner(goal, plan, results, error):
        assert error == "search failed" and set(results) == {1, 2}
        return Plan(goal=goal, steps=[
            Step(id=1, description="a"), Step(id=2, description="b"),
            Step(id=5, description="c, another way", depends_on=[1, 2]),
            Step(id=6, description="d", depends_on=[5])])

    out = run_plan("g", four_step_plan, executor, replanner)
    assert out["status"] == "done"
    assert ran == [1, 2, 3, 5, 6]  # 1 and 2 did not run twice
    assert out["results"][6] == "r6<-[5]"


def test_replans_are_capped():
    always_fails = lambda step, inputs: {"ok": False, "error": "down"}
    out = run_plan("g", four_step_plan, always_fails,
                   lambda g, p, r, e: four_step_plan(g), max_replans=2)
    assert out["status"] == "failed"


def test_a_blocked_step_ends_the_run_as_partial():
    def executor(step, inputs):
        if step.id == 3:
            return {"ok": False, "error": "x"}
        return {"ok": True, "result": step.id}
    out = run_plan("g", four_step_plan, executor,
                   lambda g, plan, r, e: plan, max_replans=1)
    assert out["status"] == "partial"


# ---------- Chapter 04: reflection ----------

def test_cheap_checks_gate_the_critic_and_issues_drive_the_revision():
    drafts, calls, critiqued = iter(["d1", "d2", "d3"]), [], []

    def generate(task, draft, issues):
        calls.append((draft, tuple(issues)))
        return next(drafts)

    checks = {"d1": ["test_empty_input failed"], "d2": [], "d3": []}
    verdicts = {"d2": Verdict(passed=False, issues=["no error case"], score=6),
                "d3": Verdict(passed=True, issues=[], score=9)}

    def critique(task, draft):
        critiqued.append(draft)
        return verdicts[draft]

    out = reflect("t", generate, critique, lambda d: checks[d])
    assert out == {"status": "passed", "draft": "d3", "attempts": 3}
    assert critiqued == ["d2", "d3"]  # d1 failed its tests: no critic call
    assert calls[1:] == [("d1", ("test_empty_input failed",)),
                         ("d2", ("no error case",))]


def test_out_of_attempts_returns_the_best_draft_for_review():
    drafts, calls = iter(["a", "b", "c", "d"]), []

    def generate(task, draft, issues):
        calls.append(draft)
        return next(drafts)

    scores = {"a": 3, "b": 7, "c": 5}
    out = reflect("t", generate,
                  lambda t, d: Verdict(passed=False, issues=[d], score=scores[d]),
                  lambda d: [], max_attempts=3)
    assert out["status"] == "needs_review" and out["draft"] == "b"
    assert len(calls) == 3  # never generated a fourth draft nobody checks


# ---------- Chapter 06: multi-agent ----------

def test_one_failing_worker_does_not_sink_the_team():
    attempts = {}

    async def supervise(task):
        return SupervisorDecision(delegations=[
            Delegation(worker=w, brief=w, expected_output="md")
            for w in ("researcher", "analyst", "writer")])

    def worker(name, failures):
        async def run(brief, expected_output):
            attempts[name] = attempts.get(name, 0) + 1
            if attempts[name] <= failures:
                raise RuntimeError("boom")
            return f"{name} done"
        return run

    async def synthesize(task, done, missing):
        return {"done": sorted(d["worker"] for d in done), "missing": missing}

    out = asyncio.run(run_team("t", supervise, {
        "researcher": worker("researcher", 0),
        "analyst": worker("analyst", 1),     # fails once, then succeeds
        "writer": worker("writer", 5)}, synthesize))
    assert out == {"done": ["analyst", "researcher"], "missing": ["writer"]}
    assert attempts == {"researcher": 1, "analyst": 2, "writer": 2}
