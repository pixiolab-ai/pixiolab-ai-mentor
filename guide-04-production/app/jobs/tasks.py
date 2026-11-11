import logging
import os

import httpx
from celery.signals import task_failure

from app.cost.budget import check_budget
from app.jobs.celery_app import app
from app.jobs.reports import fetch_signup_metrics, write_summary
from app.jobs.runs import business_today, claim_run, set_run_status
from app.notify import slack_post
from app.paging import page

log = logging.getLogger(__name__)


@app.task(name="reports.daily_signups", autoretry_for=(Exception,),
          max_retries=3, retry_backoff=60, retry_backoff_max=900,
          retry_jitter=False,  # predictable waits: 60 s, 120 s, 240 s
          time_limit=600)      # shorter than the 15-minute claim lease
def daily_signups() -> str:
    job, day = "reports.daily_signups", business_today()
    if not claim_run(job, day):
        return "skipped: already sent or in progress"
    try:
        text = write_summary(fetch_signup_metrics(day))
        slack_post(text, channel="#daily-reports")
    except Exception:
        set_run_status(job, day, "failed")  # lets the retry claim it
        raise
    set_run_status(job, day, "sent")
    ping_heartbeat("HEARTBEAT_URL_DAILY_SIGNUPS")
    return "sent"


@app.task(name="alerts.check_budget")
def check_budget_task() -> str:
    return check_budget()  # Feature 02, now on a schedule


SCHEDULED = {"reports.daily_signups", "alerts.check_budget",
             "watchdog.check_jobs"}


@task_failure.connect
def page_on_failure(sender=None, exception=None, **kwargs):
    """Fires once, after the last retry. Filtered to scheduled jobs so
    the queue's model calls (Feature 05) don't page anyone."""
    if sender is not None and sender.name in SCHEDULED:
        page(f"{sender.name} failed after retries", repr(exception))


def ping_heartbeat(env_var: str) -> None:
    """Tell an outside monitor the job ran. It alerts when pings stop,
    which also catches a dead Beat that no in-process check can see."""
    url = os.getenv(env_var)
    if not url:
        return
    try:
        httpx.get(url, timeout=5)
    except httpx.HTTPError:
        log.warning("heartbeat ping failed: %s", env_var)
