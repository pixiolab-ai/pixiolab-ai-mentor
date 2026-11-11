import os
from uuid import UUID

import psycopg
import redis
from fastapi import Depends, FastAPI, HTTPException
from prometheus_client import Gauge, make_asgi_app
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.auth import current_user
from app.config import DATABASE_URL, REDIS_URL
from app.queue.worker import run_job

api = FastAPI()
api.mount("/metrics", make_asgi_app())  # Prometheus scrapes this
r = redis.Redis.from_url(REDIS_URL)
MAX_QUEUE = int(os.getenv("MAX_QUEUE_DEPTH", "500"))
QUEUE_DEPTH = Gauge("llm_queue_depth", "Jobs waiting in the llm queue")
QUEUE_DEPTH.set_function(lambda: r.llen("llm"))  # read at scrape time


class GenerateRequest(BaseModel):  # bounded: cost is set by the client
    prompt: str = Field(min_length=1, max_length=20_000)
    max_tokens: int = Field(default=1024, ge=1, le=4096)


def enqueue(owner_id: str, kind: str, payload: dict) -> dict:
    if r.llen("llm") > MAX_QUEUE:  # shed load before Redis fills up
        raise HTTPException(503, "Busy, try again shortly",
                            headers={"Retry-After": "30"})
    with psycopg.connect(DATABASE_URL) as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (owner_id, kind, payload)"
            " VALUES (%s, %s, %s) RETURNING id",
            (owner_id, kind, Jsonb(payload))).fetchone()[0]
    run_job.delay(str(job_id))  # after the commit, so the row exists
    return {"job_id": str(job_id), "status_url": f"/jobs/{job_id}"}


@api.post("/jobs", status_code=202)
def create_job(req: GenerateRequest, user: dict = Depends(current_user)):
    return enqueue(user["id"], "generate", req.model_dump())


@api.get("/jobs/{job_id}")
def get_job(job_id: UUID, user: dict = Depends(current_user)):
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT status, result, error FROM jobs"
            " WHERE id = %s AND owner_id = %s",
            (job_id, user["id"])).fetchone()
    if row is None:  # unknown, or someone else's: same answer
        raise HTTPException(404, "No such job")
    return dict(zip(("status", "result", "error"), row))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


@api.post("/ask", status_code=202)
def ask(body: AskRequest, user: dict = Depends(current_user)):
    # Groups come from the verified token, captured right now. Nothing
    # the client sends can change them.
    return enqueue(user["id"], "ask",
                   {"question": body.question, "groups": user["groups"]})
