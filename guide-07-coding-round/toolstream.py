import json

def collect_tool_calls(events) -> list[dict]:
    """Rebuild tool calls from streamed Messages API events (dicts)."""
    pending, calls = {}, []
    for ev in events:
        kind, i = ev["type"], ev.get("index")
        if kind == "content_block_start":
            block = ev["content_block"]
            if block["type"] == "tool_use":
                pending[i] = {"id": block["id"], "name": block["name"],
                              "parts": []}
        elif kind == "content_block_delta" and i in pending:
            if ev["delta"]["type"] == "input_json_delta":
                pending[i]["parts"].append(ev["delta"]["partial_json"])
        elif kind == "content_block_stop" and i in pending:
            call = pending.pop(i)
            raw = "".join(call.pop("parts")) or "{}"
            call["input"] = json.loads(raw)  # parse only when complete
            calls.append(call)
        elif kind == "message_delta":
            if ev["delta"].get("stop_reason") == "max_tokens":
                raise ValueError("truncated: raise max_tokens and retry")
    if pending:
        raise ValueError("stream ended inside a tool call")
    return calls
