"""Feature 04: atomic run claims, the daily report, failure paging and the watchdog.

Celery tasks run in-process (eagerly, or by calling the task function), and
Slack, the model and email are replaced with recorders.
"""
import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace as NS

import psycopg
import pytest

import app.jobs.tasks as tasks_mod
import app.jobs.watchdog as watchdog_mod
from app.config import BUSINESS_TZ, DATABASE_URL
from app.jobs.celery_app import app as celery_app
from app.jobs.reports import fetch_signup_metrics
from app.jobs.runs import claim_run, set_run_status


def sql(statement: str, args=None):
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(statement, args)
        return cur.fetchall() if cur.description else None


def test_failure_pages_only_for_scheduled_jobs(monkeypatch):
    paged = []
    monkeypatch.setattr(tasks_mod, "page", lambda subject, body: paged.append(subject))
    tasks_mod.page_on_failure(sender=NS(name="llm.run_job"), exception=ValueError())
    tasks_mod.page_on_failure(sender=NS(name="reports.daily_signups"),
                              exception=ValueError())
    assert paged == ["reports.daily_signups failed after retries"]


def test_beat_schedule_runs_in_business_time():
    assert celery_app.conf.timezone == BUSINESS_TZ.key
    schedule = celery_app.conf.beat_schedule
    assert schedule["daily-signups-report"]["task"] == "reports.daily_signups"
    assert schedule["budget-check"]["task"] == "alerts.check_budget"


@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
class TestRunClaims:
    def test_claim_lifecycle(self):
        day = date(2026, 1, 1)
        assert claim_run("demo", day) is True    # first claim wins
        assert claim_run("demo", day) is False   # running: hands off
        set_run_status("demo", day, "failed")
        assert claim_run("demo", day) is True    # a failed run is reclaimed
        set_run_status("demo", day, "sent")
        assert claim_run("demo", day) is False   # sent: never again
        sql("UPDATE job_runs SET status = 'running',"
            " updated_at = now() - interval '20 minutes'")
        assert claim_run("demo", day) is True    # its worker died
        assert sql("SELECT attempts FROM job_runs") == [(3,)]

    def test_simultaneous_claims_have_one_winner(self):
        day, results, start = date(2026, 1, 2), [], threading.Barrier(8)

        def claim():
            start.wait()
            results.append(claim_run("race", day))

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(results) == 1


@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
class TestDailyReport:
    def test_metrics_are_cut_at_business_midnight(self):
        today = datetime(2026, 3, 10, tzinfo=BUSINESS_TZ)
        for _ in range(14):    # yesterday, 23:00 business time
            sql("INSERT INTO signups (created_at) VALUES (%s)",
                (today - timedelta(hours=1),))
        for days_ago in range(2, 9):
            for _ in range(7):  # the 7 days before: 7 a day
                sql("INSERT INTO signups (created_at) VALUES (%s)",
                    (today - timedelta(days=days_ago, hours=-12),))
        sql("INSERT INTO signups (created_at) VALUES (%s)",
            (today + timedelta(hours=1),))  # today: excluded
        assert fetch_signup_metrics(date(2026, 3, 10)) == {
            "signups_yesterday": 14, "daily_avg_prior_7_days": 7.0,
            "change_vs_avg": "+100%"}

    def test_second_trigger_on_the_same_day_is_skipped(self, monkeypatch):
        posted = []  # Task.apply() runs the task here, as a worker would
        monkeypatch.setattr(tasks_mod, "fetch_signup_metrics", lambda day: {})
        monkeypatch.setattr(tasks_mod, "write_summary", lambda m: "summary")
        monkeypatch.setattr(tasks_mod, "slack_post",
                            lambda text, channel: posted.append(text))
        assert tasks_mod.daily_signups.apply().get() == "sent"
        assert tasks_mod.daily_signups.apply().get().startswith("skipped")
        assert posted == ["summary"]

    def test_failed_delivery_marks_the_run_failed_for_the_retry(
            self, monkeypatch):
        def slack_down(text, channel):
            raise RuntimeError("slack down")
        monkeypatch.setattr(tasks_mod, "fetch_signup_metrics", lambda day: {})
        monkeypatch.setattr(tasks_mod, "write_summary", lambda m: "summary")
        monkeypatch.setattr(tasks_mod, "slack_post", slack_down)
        with pytest.raises(RuntimeError):
            tasks_mod.daily_signups.run()  # the task body, without retries
        assert sql("SELECT status FROM job_runs") == [("failed",)]


@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
def test_watchdog_pages_once_for_a_run_that_never_happened(monkeypatch):
    paged = []
    monkeypatch.setattr(watchdog_mod, "page",
                        lambda subject, body: paged.append(subject))

    class TenAM:  # after the 09:00 deadline, and nothing was sent
        @staticmethod
        def now(tz):
            return datetime(2026, 5, 5, 10, 0, tzinfo=tz)

    monkeypatch.setattr(watchdog_mod, "datetime", TenAM)
    assert watchdog_mod.check_jobs.run() == ["reports.daily_signups"]
    assert watchdog_mod.check_jobs.run() == []  # already paged today
    assert paged == ["reports.daily_signups has not run today"]
