"""Chapter 03, offline: watch a plan fail, get revised, and finish.

The planner, executor and replanner are plain functions here, so you can see
the control flow without a model. In a real agent the planner and replanner
are model calls that return a Plan (structured output), and each step's
executor is a short tool-use loop.

Run from guide-02-agents/:  python -m examples.planning_demo
"""
from agent.planning import Plan, Step, run_plan

GOAL = "Write a one-page brief on our three largest competitors"


def planner(goal: str) -> Plan:
    return Plan(goal=goal, steps=[
        Step(id=1, description="List the three largest competitors"),
        Step(id=2, description="Pull each one's pricing page",
             tool="web_fetch", depends_on=[1]),
        Step(id=3, description="Pull each one's latest annual report",
             tool="web_fetch", depends_on=[1]),
        Step(id=4, description="Write the brief", depends_on=[2, 3]),
    ])


def executor(step: Step, inputs: dict) -> dict:
    print(f"  run step {step.id}: {step.description}  (inputs: {sorted(inputs)})")
    if step.id == 3:  # this source is down
        return {"ok": False, "error": "annual reports site timed out"}
    return {"ok": True, "result": f"output of step {step.id}"}


def replanner(goal: str, plan: Plan, results: dict, error: str) -> Plan:
    print(f"  re-plan after: {error}")
    # Finished steps keep their IDs; replacement steps get new ones.
    return Plan(goal=goal, steps=[
        *[s for s in plan.steps if s.id in results],
        Step(id=5, description="Use each competitor's investor-relations "
                               "summary instead", tool="web_fetch",
             depends_on=[1]),
        Step(id=6, description="Write the brief", depends_on=[2, 5]),
    ])


def main() -> dict:
    print(f"goal: {GOAL}")
    outcome = run_plan(GOAL, planner, executor, replanner)
    print(f"status: {outcome['status']}, finished steps: "
          f"{sorted(outcome['results'])}")
    return outcome


if __name__ == "__main__":
    main()
