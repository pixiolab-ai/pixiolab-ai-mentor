"""Feature 05: the job API, the worker's claim, retries over minutes, the limiter.

Redis is in memory (fakeredis), the model call is a fake, and the worker task
runs in-process by calling it directly, so you can follow one job from POST
to done without starting Celery.
"""
import psycopg
import pytest

import app.queue.limiter as limiter
import app.queue.worker as worker
from app.config import DATABASE_URL
from app.llm.client import LLMResult, LLMUnavailable


def sql(statement: str, args=None):
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(statement, args)
        return cur.fetchall() if cur.description else None


def new_job(kind="generate", payload='{"prompt": "x", "max_tokens": 5}',
            **columns) -> str:
    names = ", ".join(["owner_id", "kind", "payload", *columns])
    marks = ", ".join(["%s"] * (3 + len(columns)))
    return str(sql(f"INSERT INTO jobs ({names}) VALUES ({marks}) RETURNING id",
                   ("alice", kind, payload, *columns.values()))[0][0])


def answer(text="hello"):
    return lambda user_id, feature, messages, **options: LLMResult(
        text, "claude-sonnet-5-5", False, 10, 5)


# ---------- Step 4: one rate limit shared by every worker ----------

def test_limiter_shares_one_budget_per_minute(fake_redis):
    assert limiter.take("requests", 1, per_minute=2) == 0
    assert limiter.take("requests", 1, per_minute=2) == 0
    wait = limiter.take("requests", 1, per_minute=2)
    assert 0 < wait <= 60  # seconds until the next window


def test_refused_reservation_is_given_back(fake_redis):
    assert limiter.take("tokens", 500, per_minute=1000) == 0
    assert limiter.take("tokens", 600, per_minute=1000) > 0  # would be 1,100
    assert limiter.take("tokens", 500, per_minute=1000) == 0  # 600 returned


# ---------- Steps 1-3 and 5: the API and the worker ----------

@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
class TestJobFlow:
    def test_one_job_from_post_to_done(self, api, idp, monkeypatch):
        alice, bob = idp.header(sub="alice"), idp.header(sub="bob")
        assert api.post("/jobs", json={"prompt": "hi"}).status_code == 401

        r = api.post("/jobs", json={"prompt": "A haiku about queues"},
                     headers=alice)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert api.queued == [job_id]  # queued after the row was committed

        status = api.get(f"/jobs/{job_id}", headers=alice).json()
        assert status["status"] == "queued"
        # Someone else's job looks exactly like a job that doesn't exist.
        assert api.get(f"/jobs/{job_id}", headers=bob).status_code == 404

        monkeypatch.setattr(worker, "tracked_generate", answer("hello"))
        worker.run_job.run(job_id)  # what a Celery worker would do
        assert api.get(f"/jobs/{job_id}", headers=alice).json() == {
            "status": "done", "error": None,
            "result": {"text": "hello", "model": "claude-sonnet-5-5",
                       "fallback": False}}

        worker.run_job.run(job_id)  # a redelivered message: no second run
        assert sql("SELECT attempts FROM jobs") == [(1,)]

    def test_requests_are_bounded_before_they_cost_anything(self, api, idp):
        too_big = {"prompt": "hi", "max_tokens": 100_000}
        assert api.post("/jobs", json=too_big,
                        headers=idp.header()).status_code == 422

    def test_a_deep_queue_sheds_load(self, api, idp, fake_redis, monkeypatch):
        import app.api as api_mod
        monkeypatch.setattr(api_mod, "MAX_QUEUE", 0)
        fake_redis.rpush("llm", "a-waiting-job")
        r = api.post("/jobs", json={"prompt": "hi"}, headers=idp.header())
        assert r.status_code == 503 and r.headers["retry-after"] == "30"
        assert "llm_queue_depth 1.0" in api.get("/metrics/").text

    def test_providers_down_requeues_with_the_suggested_wait(
            self, fake_redis, monkeypatch):
        job_id = new_job()

        def down(*args, **kwargs):
            raise LLMUnavailable(retry_after=12)

        class Retry(Exception):
            pass

        asked = {}
        monkeypatch.setattr(worker, "tracked_generate", down)
        monkeypatch.setattr(worker.run_job, "retry",
                            lambda **kw: asked.update(kw) or Retry())
        with pytest.raises(Retry):
            worker.run_job.run(job_id)
        assert asked == {"countdown": 12}
        assert sql("SELECT status FROM jobs") == [("queued",)]

    def test_a_bug_fails_the_job_instead_of_looping(self, fake_redis):
        job_id = new_job(kind="no-such-handler", payload="{}")
        worker.run_job.run(job_id)
        status, error = sql("SELECT status, error FROM jobs")[0]
        assert status == "failed" and "KeyError" in error

    def test_over_the_rate_limit_waits_without_counting_a_retry(
            self, fake_redis, monkeypatch):
        job_id = new_job()
        monkeypatch.setattr(worker, "wait_needed", lambda tokens: 5.0)
        later = []
        monkeypatch.setattr(worker.run_job, "apply_async",
                            lambda args, countdown: later.append((args, countdown)))
        worker.run_job.run(job_id)
        assert later == [((job_id,), 5.0)]
        assert sql("SELECT status FROM jobs") == [("queued",)]

    def test_running_elsewhere_checks_back_then_takes_over_a_dead_lease(
            self, fake_redis, monkeypatch):
        job_id = new_job(status="running")
        later = []
        monkeypatch.setattr(worker.run_job, "apply_async",
                            lambda args, countdown: later.append(countdown))
        worker.run_job.run(job_id)
        assert later == [60]  # still running: look again in a minute

        sql("UPDATE jobs SET updated_at = now() - interval '12 minutes'")
        monkeypatch.setattr(worker, "tracked_generate", answer("recovered"))
        worker.run_job.run(job_id)  # the lease expired: its worker died
        assert sql("SELECT status FROM jobs") == [("done",)]
