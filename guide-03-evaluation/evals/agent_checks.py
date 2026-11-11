from math import comb

def tool_calls(trace):
    return [s for s in trace["spans"] if s["kind"] == "tool"]

def trajectory_violations(trace, max_searches=5, refund_limit=200):
    """Rules over one trace. Returns the names of the rules it broke."""
    calls = tool_calls(trace)
    names = [c["name"] for c in calls]
    broken = []
    if names.count("search_kb") > max_searches:
        broken.append("too_many_searches")
    for i, call in enumerate(calls):
        before = calls[:i]
        if call["name"] == "issue_refund":
            if not any(c["name"] == "lookup_order" for c in before):
                broken.append("refund_before_lookup")
            approved = any(c["name"] == "request_approval"
                           and c.get("result", {}).get("approved")
                           for c in before)
            if call["args"]["amount"] > refund_limit and not approved:
                broken.append("refund_over_limit_unapproved")
        if call["name"] == "delete_account" and "confirm" not in names[:i]:
            broken.append("delete_without_confirmation")
    return broken

def pass_at_k(n, c, k):
    """P(at least one of k trials succeeds), given c successes in n."""
    return 1 - comb(n - c, k) / comb(n, k)

def pass_hat_k(n, c, k):
    """P(all k trials succeed), given c successes in n trials."""
    return comb(c, k) / comb(n, k)

# A task that passed 8 of 10 trials:
# pass_at_k(10, 8, 4) == 1.0, but pass_hat_k(10, 8, 4) is about 0.33