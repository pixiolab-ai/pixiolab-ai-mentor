import psycopg
from psycopg.types.json import Jsonb

from app.config import DATABASE_URL
from app.cost.tracking import tracked_generate
from app.jobs.celery_app import app
from app.llm.client import LLMUnavailable
from app.queue.limiter import wait_needed

HANDLERS = {}  # job kind -> function(owner_id, payload) -> result dict


def handler(kind: str):
    def register(fn):
        HANDLERS[kind] = fn
        return fn
    return register


@handler("generate")
def handle_generate(owner_id: str, payload: dict) -> dict:
    r = tracked_generate(owner_id, "generate",
                         [{"role": "user", "content": payload["prompt"]}],
                         max_tokens=payload["max_tokens"])
    return {"text": r.text, "model": r.model, "fallback": r.fallback}


# At-least-once delivery: let exactly one worker in. The lease (11 min)
# outlives the hard time limit (10 min), so an older "running" row
# belongs to a worker that died.
CLAIM = """
UPDATE jobs SET status = 'running', attempts = attempts + 1,
                updated_at = now()
 WHERE id = %s
   AND (status = 'queued' OR (status = 'running'
        AND updated_at < now() - interval '11 minutes'))
RETURNING owner_id, kind, payload
"""


def set_job(job_id: str, status: str, result=None, error=None) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE jobs SET status = %s, result = %s, error = %s,"
            " updated_at = now() WHERE id = %s",
            (status, Jsonb(result) if result else None, error, job_id))


def job_status(job_id: str) -> str | None:
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = %s",
                           (job_id,)).fetchone()
    return row[0] if row else None


@app.task(name="llm.run_job", bind=True, max_retries=5)
def run_job(self, job_id: str) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(CLAIM, (job_id,)).fetchone()
    if row is None:  # finished, or another worker holds the lease
        if job_status(job_id) == "running":  # check again until it ends
            run_job.apply_async((job_id,), countdown=60)
        return
    owner_id, kind, payload = row
    # Rough input estimate: ~4 characters a token, plus retrieved context.
    wait = wait_needed(len(str(payload)) // 4 + 2_000)
    if wait:  # over the shared limit: requeue without counting a retry
        set_job(job_id, "queued")
        run_job.apply_async((job_id,), countdown=wait)
        return
    try:
        result = HANDLERS[kind](owner_id, payload)
    except LLMUnavailable as exc:   # both providers down: retry later
        if self.request.retries >= self.max_retries:
            set_job(job_id, "failed", error="model providers unavailable")
            return
        set_job(job_id, "queued")
        delay = exc.retry_after or 30 * 2 ** self.request.retries
        raise self.retry(countdown=min(delay, 600))  # < visibility timeout
    except Exception as exc:        # a bug or a refusal: don't loop on it
        set_job(job_id, "failed", error=repr(exc)[:500])
        return
    set_job(job_id, "done", result=result)
