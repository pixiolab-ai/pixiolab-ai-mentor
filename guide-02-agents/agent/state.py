import sqlite3

from pydantic import BaseModel

class AgentState(BaseModel):
    schema_version: int = 2
    run_id: str
    step: int = 0
    messages: list[dict] = []        # conversation state (blocks as dicts)
    plan: dict | None = None         # task state
    results: dict[str, str] = {}
    tokens_used: int = 0
    awaiting_human: bool = False     # added in v2; v1 rows get the default
    final: str | None = None

db = sqlite3.connect("runs.db")
db.execute("CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT, "
           "step INTEGER, state TEXT, PRIMARY KEY (run_id, step))")

def checkpoint(state: AgentState) -> None:
    with db:                                       # commit or roll back
        db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                   (state.run_id, state.step, state.model_dump_json()))

def resume(run_id: str) -> AgentState | None:
    row = db.execute("SELECT state FROM checkpoints WHERE run_id = ? "
                     "ORDER BY step DESC LIMIT 1", (run_id,)).fetchone()
    return AgentState.model_validate_json(row[0]) if row else None
