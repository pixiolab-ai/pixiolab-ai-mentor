from datetime import date, datetime

import psycopg

from app.config import BUSINESS_TZ, DATABASE_URL

# One statement; the primary key serializes racing workers. A run can be
# reclaimed only if it failed, or if its worker died (lease expired).
CLAIM = """
INSERT INTO job_runs (job, run_date, status) VALUES (%s, %s, 'running')
ON CONFLICT (job, run_date) DO UPDATE
   SET status = 'running', attempts = job_runs.attempts + 1,
       updated_at = now()
 WHERE job_runs.status = 'failed'
    OR (job_runs.status = 'running'
        AND job_runs.updated_at < now() - interval '15 minutes')
RETURNING attempts
"""


def business_today() -> date:
    return datetime.now(BUSINESS_TZ).date()


def claim_run(job: str, day: date) -> bool:
    """True if this worker now owns the run. False: sent, or in progress."""
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute(CLAIM, (job, day)).fetchone() is not None


def set_run_status(job: str, day: date, status: str) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE job_runs SET status = %s, updated_at = now()"
            " WHERE job = %s AND run_date = %s", (status, job, day))
