from typing import Literal
from pydantic import BaseModel

class Step(BaseModel):
    id: int
    description: str               # concrete, with a checkable output
    tool: str | None = None
    depends_on: list[int] = []
    status: Literal["pending", "done", "failed"] = "pending"

class Plan(BaseModel):
    goal: str
    steps: list[Step]

def next_ready(plan: Plan, results: dict) -> Step | None:
    ready = (s for s in plan.steps if s.status == "pending"
             and all(d in results for d in s.depends_on))
    return next(ready, None)

def run_plan(goal, planner, executor, replanner, max_replans=2):
    plan, results, replans = planner(goal), {}, 0     # planner -> Plan
    while (step := next_ready(plan, results)) is not None:
        outcome = executor(step, {d: results[d] for d in step.depends_on})
        if outcome["ok"]:
            step.status = "done"
            results[step.id] = outcome["result"]
            continue
        step.status = "failed"
        if replans == max_replans:
            return {"status": "failed", "plan": plan, "results": results}
        replans += 1
        plan = replanner(goal, plan, results, outcome["error"])
        for s in plan.steps:                    # finished work stays done
            s.status = "done" if s.id in results else s.status
    left = [s.id for s in plan.steps if s.status != "done"]
    return {"status": "partial" if left else "done",
            "plan": plan, "results": results}
