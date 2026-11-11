import json
from datetime import date, datetime, time, timedelta

import psycopg

from app.config import BUSINESS_TZ, DATABASE_URL
from app.cost.tracking import tracked_generate


def fetch_signup_metrics(day: date) -> dict:
    """Yesterday vs the 7 days before it, cut at business midnight."""
    today = datetime.combine(day, time.min, tzinfo=BUSINESS_TZ)
    yday, week = today - timedelta(days=1), today - timedelta(days=8)
    with psycopg.connect(DATABASE_URL) as conn:
        yesterday, prior = conn.execute(
            "SELECT count(*) FILTER (WHERE created_at >= %(y)s),"
            "       count(*) FILTER (WHERE created_at < %(y)s)"
            " FROM signups"
            " WHERE created_at >= %(w)s AND created_at < %(t)s",
            {"y": yday, "w": week, "t": today}).fetchone()
    avg = prior / 7
    change = (yesterday - avg) / avg if avg else None
    return {"signups_yesterday": yesterday,
            "daily_avg_prior_7_days": round(avg, 1),
            "change_vs_avg": f"{change:+.0%}" if change is not None else "n/a"}


REPORT_SYSTEM = (
    "You write short daily reports for a Slack channel. At most three "
    "bullet points. Lead with the most important change. Use only the "
    "numbers provided and never calculate new ones.")


def write_summary(metrics: dict) -> str:
    r = tracked_generate(
        "system:scheduler", "daily_report",
        [{"role": "user", "content": json.dumps(metrics)}],
        system=REPORT_SYSTEM, max_tokens=2048)
    return r.text
