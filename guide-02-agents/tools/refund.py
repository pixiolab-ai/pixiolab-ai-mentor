import hashlib
import json

def idempotency_key(run_id: str, step: int, tool: str, args: dict) -> str:
    """Same run, step, tool and arguments give the same key on retry."""
    raw = json.dumps([run_id, step, tool, args], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def issue_refund(payments, run_id: str, step: int, args: dict) -> dict:
    key = idempotency_key(run_id, step, "issue_refund", args)
    # The payment API stores the key; a second call with it is a no-op.
    return payments.refund(order_id=args["order_id"],
                           amount_cents=args["amount_cents"],
                           idempotency_key=key)
