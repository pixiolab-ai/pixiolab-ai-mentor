def check_trajectory(events, max_searches=5):
    """events: ordered dicts, e.g. {"kind": "tool", "name": "refund",
    "approved": True} or {"kind": "answer"}. Returns rule violations."""
    problems, searches = [], 0
    for e in events:
        if e["kind"] == "tool" and e["name"] == "search_docs":
            searches += 1
        is_refund = e["kind"] == "tool" and e["name"] == "refund"
        if is_refund and not e.get("approved"):
            problems.append("refund issued without approval")
        if e["kind"] == "answer" and searches == 0:
            problems.append("answered before retrieving")
    if searches > max_searches:
        problems.append(f"{searches} searches (limit {max_searches})")
    return problems
