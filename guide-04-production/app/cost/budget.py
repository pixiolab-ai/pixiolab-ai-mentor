from datetime import datetime
from decimal import Decimal

import psycopg

from app.config import BUSINESS_TZ, DAILY_BUDGET_USD, DATABASE_URL
from app.notify import slack_post

ALERT_AT = Decimal("0.8")  # warn at 80%, while there's time to act


def check_budget() -> str:
    """Alert once per business day when spend passes 80% of the budget."""
    now = datetime.now(BUSINESS_TZ)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    with psycopg.connect(DATABASE_URL) as conn:
        spent = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage"
            " WHERE created_at >= %s", (start,)).fetchone()[0]
        if spent < DAILY_BUDGET_USD * ALERT_AT:
            return "ok"
        # First writer wins: the primary key turns a second insert for the
        # same day into a no-op, even when two checks run at once.
        first = conn.execute(
            "INSERT INTO alerts_sent (kind, day) VALUES ('budget', %s)"
            " ON CONFLICT DO NOTHING RETURNING 1", (now.date(),)).fetchone()
        if first is None:
            return "already alerted today"
        share = spent / DAILY_BUDGET_USD
        slack_post(f"LLM spend today is ${spent:,.2f}, {share:.0%} of the "
                   f"${DAILY_BUDGET_USD:,.2f} daily budget.",
                   channel="#llm-alerts")
    return "alerted"  # the block commits on exit; an exception rolls back
