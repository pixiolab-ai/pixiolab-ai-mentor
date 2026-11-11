from datetime import datetime

import psycopg

from app.config import BUSINESS_TZ, DATABASE_URL
from app.jobs.celery_app import app
from app.paging import page

DEADLINES = {"reports.daily_signups": 9}  # job -> hour it must be sent by

SENT = ("SELECT 1 FROM job_runs WHERE job = %s AND run_date = %s"
        " AND status = 'sent'")
FIRST_ALERT = ("INSERT INTO alerts_sent (kind, day) VALUES (%s, %s)"
               " ON CONFLICT DO NOTHING RETURNING 1")


@app.task(name="watchdog.check_jobs")
def check_jobs() -> list[str]:
    """Catch jobs that never ran. Pages once per job per day."""
    now = datetime.now(BUSINESS_TZ)
    late = []
    for job, hour in DEADLINES.items():
        if now.hour < hour:
            continue
        with psycopg.connect(DATABASE_URL) as conn:
            if conn.execute(SENT, (job, now.date())).fetchone():
                continue
            if conn.execute(FIRST_ALERT,
                            (f"late:{job}", now.date())).fetchone():
                page(f"{job} has not run today",
                     f"It was due by {hour}:00 {BUSINESS_TZ.key}.")
                late.append(job)
    return late
