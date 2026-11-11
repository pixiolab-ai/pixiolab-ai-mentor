from celery import Celery
from celery.schedules import crontab

from app.config import BUSINESS_TZ, REDIS_URL

app = Celery("app", broker=REDIS_URL, include=[
    "app.jobs.tasks", "app.jobs.watchdog",   # Feature 04
    "app.queue.worker", "app.rag.answer",    # Features 05 and 06
])
app.conf.update(
    timezone=BUSINESS_TZ.key,
    task_ignore_result=True,          # job state lives in Postgres
    task_routes={"llm.*": {"queue": "llm"}},  # model jobs: own workers
    task_acks_late=True,              # ack when the task ends, not starts
    task_reject_on_worker_lost=True,  # requeue if the process dies
    worker_prefetch_multiplier=1,     # each process holds one job
    task_time_limit=600,              # hard limit: kill after 10 min
    # Redis redelivers unacked jobs after this. Keep it above the time
    # limit and above any retry countdown, or jobs run twice.
    broker_transport_options={"visibility_timeout": 900},
)
app.conf.beat_schedule = {
    "daily-signups-report": {"task": "reports.daily_signups",
                             "schedule": crontab(hour=8, minute=0)},
    "budget-check": {"task": "alerts.check_budget",  # Feature 02
                     "schedule": crontab(minute="*/15")},
    "job-watchdog": {"task": "watchdog.check_jobs",
                     "schedule": crontab(minute="*/30")},
}
