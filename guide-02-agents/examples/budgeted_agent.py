"""Chapter 08 wired into Chapter 01's loop: budgets and repeat detection in code.

Chapter 08 says: call charge() after each model call, check exceeded() before
the next, and answer a repeated call with an error result instead of running
it. This is that loop. It is not printed in the guide; agent/loop.py and
agent/budget.py are.

Run from guide-02-agents/ (needs ANTHROPIC_API_KEY):

    python -m examples.budgeted_agent "Where is my order ORD-12345678?"
"""
import json
import sys

from agent import loop
from agent.budget import Budget
from agent.tools import execute
from examples.support_agent import register_tools

# ILLUSTRATIVE prices in USD per million tokens, used only to charge the
# budget. Copy current values from your provider's pricing page.
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}
REPEATED = ("You already ran this exact call twice; try another approach "
            "or answer with what you have.")


def call_cost(usage) -> float:
    return (usage.input_tokens * PRICE_PER_MTOK["input"]
            + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000


def run_budgeted_agent(task, tools, execute, budget: Budget | None = None):
    budget = budget or Budget()
    messages = [{"role": "user", "content": task}]
    while (limit := budget.exceeded()) is None:  # check before each call
        resp = loop.client.messages.create(
            model=loop.MODEL, max_tokens=16000, tools=tools, messages=messages)
        u = resp.usage
        budget.charge(u.input_tokens + u.output_tokens, call_cost(u))
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"status": resp.stop_reason, "answer": text,
                    "used": budget.used}
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if budget.repeated(block.name, block.input):
                out = {"ok": False, "error": REPEATED}  # don't run it again
            else:
                out = execute(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out),
                            "is_error": not out.get("ok", False)})
        messages.append({"role": "user", "content": results})
    # A limit is a partial result with a flag, not a crash.
    return {"status": f"budget_exceeded:{limit}", "answer": None,
            "used": budget.used}


def main(argv: list[str]) -> dict:
    task = argv[0] if argv else "Where is my order ORD-12345678?"
    budget = Budget(max_steps=6, max_tokens=60_000, max_seconds=60,
                    max_cost_usd=0.25)
    result = run_budgeted_agent(task, register_tools(), execute, budget)
    print(f"[{result['status']}] used: {result['used']}\n{result['answer']}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
