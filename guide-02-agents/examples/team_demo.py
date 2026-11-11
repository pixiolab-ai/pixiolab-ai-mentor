"""Chapter 06, offline: parallel workers, one flaky and one broken.

The supervisor, workers and synthesizer are async stand-ins for model calls.
The analyst fails once and succeeds on retry; the writer always fails, so it
is reported as missing instead of sinking the whole run.

Run from guide-02-agents/:  python -m examples.team_demo
"""
import asyncio

from agent.team import Delegation, SupervisorDecision, run_team

calls: dict[str, int] = {}


async def supervise(task: str) -> SupervisorDecision:
    return SupervisorDecision(delegations=[
        Delegation(worker="researcher", brief=f"Find sources for: {task}",
                   expected_output="bullet list of sources with URLs"),
        Delegation(worker="analyst", brief=f"Compare options for: {task}",
                   expected_output="a table with one row per option"),
        Delegation(worker="writer", brief=f"Draft an intro for: {task}",
                   expected_output="two paragraphs of markdown"),
    ])


def worker(name: str, failures: int):
    async def run(brief: str, expected_output: str) -> str:
        calls[name] = calls.get(name, 0) + 1
        await asyncio.sleep(0.05)  # stands in for a model call
        if calls[name] <= failures:
            raise TimeoutError(f"{name} timed out")
        return f"{name}: {expected_output}"
    return run


async def synthesize(task: str, done: list[dict], missing: list[str]) -> dict:
    return {"task": task, "parts": [d["worker"] for d in done],
            "missing": missing}


def main() -> dict:
    workers = {"researcher": worker("researcher", failures=0),
               "analyst": worker("analyst", failures=1),
               "writer": worker("writer", failures=99)}
    report = asyncio.run(run_team("vector databases for a 10M-doc corpus",
                                  supervise, workers, synthesize))
    print(report, "\ncalls per worker:", calls)
    return report


if __name__ == "__main__":
    main()
