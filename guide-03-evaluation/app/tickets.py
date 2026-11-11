"""The feature under test in Chapter 05: summarize one support ticket.

evals/run_tickets.py calls summarize_ticket(text) and grades the dict it
returns ("category" and "summary"). Structured outputs constrain the answer
to a JSON schema, so the graders always receive the fields they check; a
refusal or a cut-off answer raises, and the harness records it as a failure.
"""
import json
import os

import anthropic

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")  # set to a current model ID
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

SYSTEM = """You summarize customer support tickets for the support team.
category: billing, account or technical.
summary: one sentence that keeps the customer's key facts, in their words."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category", "summary"],
    "properties": {
        "category": {"type": "string",
                     "enum": ["billing", "account", "technical"]},
        "summary": {"type": "string"},
    },
}


def summarize_ticket(ticket: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=2048, system=SYSTEM,
        messages=[{"role": "user", "content": ticket}],
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": SCHEMA}})
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason != "end_turn" or not text:
        raise RuntimeError(f"no usable answer (stop_reason={resp.stop_reason})")
    return json.loads(text)
