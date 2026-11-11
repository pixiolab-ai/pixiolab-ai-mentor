from pydantic import BaseModel

class Verdict(BaseModel):
    passed: bool
    issues: list[str]
    score: int                       # 1-10, from the rubric

def reflect(task, generate, critique, run_checks, max_attempts=3):
    """generate(task, draft, issues) -> str; critique(task, draft) -> Verdict;
    run_checks(draft) -> list of failures from tests, schema and lint."""
    draft, best = generate(task, None, []), None
    for attempt in range(1, max_attempts + 1):
        failures = run_checks(draft)           # cheap and deterministic
        verdict = None if failures else critique(task, draft)
        if verdict is not None and verdict.passed:
            return {"status": "passed", "draft": draft, "attempts": attempt}
        issues = failures or verdict.issues
        score = verdict.score if verdict is not None else 0
        if best is None or score > best["score"]:   # compare scores only
            best = {"score": score, "draft": draft, "issues": issues}
        if attempt == max_attempts:
            break                    # never pay for an unchecked draft
        draft = generate(task, draft, issues)
    return {"status": "needs_review", **best}
