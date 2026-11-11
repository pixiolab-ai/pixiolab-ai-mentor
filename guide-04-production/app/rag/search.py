import os

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL
from app.rag.embed import embed

READ_URL = os.getenv("RAG_DATABASE_URL", DATABASE_URL)  # as app_rag

# Filter inside the query: restricted rows never leave the database.
# Iterative scans (pgvector 0.8+) keep searching until k rows pass.
SEARCH = """
WITH hits AS MATERIALIZED (
  SELECT doc_id, content, embedding <=> %(q)s AS distance
  FROM chunks
  WHERE allowed_groups && %(groups)s::text[]
  ORDER BY distance LIMIT %(k)s
)
SELECT doc_id, content FROM hits ORDER BY distance + 0  -- "+ 0": PG 17+
"""


def search(query: str, groups: list[str], k: int = 5) -> list[tuple]:
    if not groups:
        return []  # no groups, no access
    qvec = embed([query])[0]
    with psycopg.connect(READ_URL) as conn:
        register_vector(conn)
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SELECT set_config('app.groups', %s, true)",
                     (",".join(groups),))  # read by the RLS policy
        return conn.execute(
            SEARCH, {"q": qvec, "groups": groups, "k": k}).fetchall()
