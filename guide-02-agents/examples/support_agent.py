"""Chapters 01, 02 and 09 together: a support agent on the raw tool-use loop.

- agent/loop.py runs the loop (call, run tools, return results, repeat).
- agent/tools/ validates every tool call against a Pydantic schema and turns
  failures into error results the model can read and recover from.
- agent/tools/lookup_order.py is Chapter 09's hand-written tool definition.

The two backends are toy in-memory data, so the only network call is to the
model. Run from guide-02-agents/ (needs ANTHROPIC_API_KEY):

    python -m examples.support_agent "Where is my order ORD-12345678?"
"""
import re
import sys

from pydantic import BaseModel, ConfigDict, Field

from agent.loop import run_agent
from agent.tools import TOOLS, execute, tool_definitions
from agent.tools.lookup_order import LOOKUP_ORDER

HELP_DOCS = [
    {"title": "Track a package", "url": "https://help.example.com/track",
     "snippet": "Open Orders, choose the order, then Track package."},
    {"title": "Start a return", "url": "https://help.example.com/returns",
     "snippet": "Returns are accepted within 30 days from the Orders page."},
    {"title": "Reset your password", "url": "https://help.example.com/password",
     "snippet": "Choose Forgot password on the sign-in page."},
]
ORDERS = {
    "ORD-12345678": {"status": "shipped", "items": ["TrailRunner 3 boots"],
                     "total_cents": 12900,
                     "events": ["2026-11-02 label created",
                                "2026-11-03 picked up by carrier"]},
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_docs(query: str, max_results: int) -> list[dict]:
    """Toy keyword search: rank help pages by shared words."""
    q = words(query)
    scored = [(len(q & words(d["title"] + " " + d["snippet"])), d)
              for d in HELP_DOCS]
    hits = [d for score, d in sorted(scored, key=lambda s: -s[0]) if score]
    return hits[:max_results]


class LookupOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^ORD-[0-9]{8}$")


def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:  # an error that teaches the model what to do next
        raise LookupError("Order not found. Check the order_id: ORD- "
                          "followed by 8 digits, as the customer wrote it.")
    return {"order_id": order_id, **order}


def register_tools() -> list[dict]:
    """Connect the backends to the registry and return the definitions."""
    TOOLS["search_docs"]["fn"] = search_docs  # replaces the stub
    TOOLS["lookup_order"] = {"description": LOOKUP_ORDER["description"],
                             "schema": LookupOrderInput, "fn": lookup_order}
    # search_docs's definition is generated from its Pydantic schema
    # (Chapter 02); lookup_order's is hand-written (Chapter 09). Both run
    # through the same validating execute().
    return [d for d in tool_definitions() if d["name"] != "lookup_order"] + [
        LOOKUP_ORDER]


def main(argv: list[str]) -> dict:
    task = argv[0] if argv else "Where is my order ORD-12345678?"
    result = run_agent(task, register_tools(), execute)
    print(f"[{result['status']}]\n{result['answer']}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
