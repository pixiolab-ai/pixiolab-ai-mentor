import logging

import psycopg

from app.config import DATABASE_URL, TRACK_USAGE
from app.cost.pricing import cost_usd
from app.llm.client import LLMResult, generate
from app.llm.text import NoText

log = logging.getLogger(__name__)


def tracked_generate(user_id: str, feature: str, messages: list[dict],
                     **options) -> LLMResult:
    """generate() plus one llm_usage row per call, refusals included."""
    try:
        result = generate(messages, **options)
    except NoText as exc:
        if exc.result is not None:
            record(user_id, feature, exc.result)
        raise
    record(user_id, feature, result)
    return result


INSERT = """
INSERT INTO llm_usage (user_id, feature, model, fallback, input_tokens,
    output_tokens, cache_read_tokens, cache_write_tokens, cost_usd)
VALUES (%(user_id)s, %(feature)s, %(model)s, %(fallback)s,
    %(input_tokens)s, %(output_tokens)s, %(cache_read_tokens)s,
    %(cache_write_tokens)s, %(cost_usd)s)
"""


def record(user_id: str, feature: str, r: LLMResult) -> None:
    if not TRACK_USAGE:
        return
    row = {"user_id": user_id, "feature": feature, "model": r.model,
           "fallback": r.fallback, "input_tokens": r.input_tokens,
           "output_tokens": r.output_tokens,
           "cache_read_tokens": r.cache_read_tokens,
           "cache_write_tokens": r.cache_write_tokens}
    row["cost_usd"] = cost_usd(row)
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            conn.execute(INSERT, row)  # commits when the block exits
    except psycopg.Error:
        # Never fail a call you already paid for because bookkeeping
        # failed. Log the row so it can be replayed into the table.
        log.exception("usage row not stored: %s", row)
