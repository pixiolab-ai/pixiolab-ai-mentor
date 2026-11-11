from functools import cache
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from app.cost.tracking import tracked_generate


class TicketSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["billing", "technical", "account"]
    urgency: Literal["low", "medium", "high"]
    summary: str


# Plain JSON Schema that both providers accept for structured output.
SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string",
                     "enum": ["billing", "technical", "account"]},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
    },
    "required": ["category", "urgency", "summary"],
    "additionalProperties": False,
}


@cache
def load_prompt(name: str) -> dict:
    with open(f"prompts/{name}.yaml") as f:
        return yaml.safe_load(f)


def summarize_ticket(ticket: str, user_id: str = "system",
                     skip_primary: bool = False) -> TicketSummary:
    cfg = load_prompt("ticket_summary")
    r = tracked_generate(
        user_id, "ticket_summary",
        [{"role": "user", "content": ticket}],
        system=cfg["system"], model=cfg["model"], effort=cfg["effort"],
        max_tokens=cfg["max_tokens"], json_schema=SCHEMA,
        skip_primary=skip_primary)
    return TicketSummary.model_validate_json(r.text)  # raises if invalid
