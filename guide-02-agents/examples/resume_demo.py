"""Chapter 10, offline: checkpoint every step, crash, resume from the last one.

agent/state.py opens runs.db in the current directory when it is imported,
so this demo creates guide-02-agents/runs.db (git-ignored).

Run from guide-02-agents/:  python -m examples.resume_demo
"""
from uuid import uuid4

from agent.state import AgentState, checkpoint, resume

RUN_ID = f"demo-{uuid4().hex[:8]}"  # a new run each time
STEPS = ["search the docs", "draft the answer", "check the citations",
         "send the reply"]


def work(state: AgentState, crash_at: int | None) -> AgentState:
    while state.step < len(STEPS):
        if state.step == crash_at:
            raise RuntimeError(f"worker died before step {state.step}")
        state.results[str(state.step)] = f"did: {STEPS[state.step]}"
        state.step += 1
        checkpoint(state)  # after every step
    state.final = "reply sent"
    checkpoint(state)
    return state


def main() -> AgentState:
    try:
        work(resume(RUN_ID) or AgentState(run_id=RUN_ID), crash_at=2)
    except RuntimeError as exc:
        print(f"crash: {exc}")
    state = resume(RUN_ID)
    print(f"resumed at step {state.step} with results {sorted(state.results)}")
    state = work(state, crash_at=None)
    print(f"finished: {state.final}, steps done: {state.step}")
    return state


if __name__ == "__main__":
    main()
