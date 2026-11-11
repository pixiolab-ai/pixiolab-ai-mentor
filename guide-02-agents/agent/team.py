import asyncio
from typing import Literal

from pydantic import BaseModel

class Delegation(BaseModel):
    worker: Literal["researcher", "analyst", "writer"]
    brief: str                   # self-contained: goal, context, limits
    expected_output: str         # a format the synthesizer can merge

class SupervisorDecision(BaseModel):
    delegations: list[Delegation]

async def run_team(task, supervise, workers, synthesize, retries=1):
    decision = await supervise(task)          # -> SupervisorDecision
    pending, done = list(decision.delegations), []
    for _ in range(retries + 1):
        outs = await asyncio.gather(
            *(workers[d.worker](d.brief, d.expected_output) for d in pending),
            return_exceptions=True)           # one failure keeps the rest
        failed = []
        for d, out in zip(pending, outs):
            if isinstance(out, BaseException):
                failed.append(d)
            else:
                done.append({"worker": d.worker, "output": out})
        pending = failed
        if not pending:
            break
    missing = [d.worker for d in pending]     # still failing after retries
    return await synthesize(task, done, missing)
