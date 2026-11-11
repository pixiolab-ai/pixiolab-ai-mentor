import anthropic
from pydantic import BaseModel, Field

from app.config import JUDGE_MODEL

judge_client = anthropic.Anthropic()  # SDK retries are fine offline


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


JUDGE_PROMPT = """Grade a support-ticket summary against the ticket.
<ticket>{ticket}</ticket>
<summary>{summary}</summary>
<key_facts>{facts}</key_facts>
Score 1 to 5. 5: keeps every key fact and adds nothing that is not in
the ticket. 3: misses one key fact. 1: misses most key facts or invents
one. Give a one-sentence reason."""


def judge(ticket: str, summary: str, facts: list[str]) -> Verdict:
    resp = judge_client.messages.parse(
        model=JUDGE_MODEL, max_tokens=2048,
        output_config={"effort": "low"},
        output_format=Verdict,  # structured output, validated by Pydantic
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            ticket=ticket, summary=summary, facts="; ".join(facts))}])
    if resp.parsed_output is None:  # refusal or no text
        raise ValueError(f"judge gave no verdict: {resp.stop_reason}")
    return resp.parsed_output


def score_case(case: dict, run) -> dict:
    """One trial: code checks first, then the judge. Errors are recorded
    per trial and never abort the whole run."""
    try:
        out = run(case["input"])  # validated TicketSummary
    except Exception as exc:      # API error, refusal, invalid JSON
        return {"passed": False, "judge": None, "error": repr(exc)[:300]}
    checks = {
        "category": out.category == case["expect"]["category"],
        "urgency": out.urgency == case["expect"]["urgency"],
    }
    trial = {"passed": all(checks.values()), "checks": checks,
             "summary": out.summary, "judge": None}
    try:
        verdict = judge(case["input"], out.summary, case["key_facts"])
        trial.update(judge=verdict.score, reason=verdict.reason)
    except Exception as exc:  # a judge failure is not a quality score
        trial["judge_error"] = repr(exc)[:300]
    return trial
