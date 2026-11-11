import json
import os

import anthropic

client = anthropic.Anthropic()                      # reads ANTHROPIC_API_KEY
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")  # use a current model ID

def run_agent(task, tools, execute, max_steps=10):
    """tools: tool definitions. execute(name, args) -> dict with "ok"."""
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL, max_tokens=16000, tools=tools, messages=messages)
        # Keep every block (text, thinking, tool_use) unchanged in history.
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":       # done, refused or cut off
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"status": resp.stop_reason, "answer": text}
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = execute(block.name, block.input)   # never raises
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out),
                    "is_error": not out.get("ok", False),
                })
        messages.append({"role": "user", "content": results})  # one turn
    return {"status": "max_steps", "answer": None}
