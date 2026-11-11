import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL
from app.rag.embed import embed, split_into_chunks

INSERT = ("INSERT INTO chunks (doc_id, chunk_no, content, embedding,"
          " allowed_groups) VALUES (%s, %s, %s, %s, %s)")


def ingest(doc_id: str, text: str, allowed_groups: list[str]) -> int:
    """(Re)index one document with the groups copied from its source."""
    if not allowed_groups:  # unknown permissions: never index it
        raise ValueError(f"{doc_id} has no permissions, refusing to index")
    chunks = split_into_chunks(text)
    rows = [(doc_id, i, chunk, vec, allowed_groups)
            for i, (chunk, vec) in enumerate(zip(chunks, embed(chunks)))]
    with psycopg.connect(DATABASE_URL) as conn:  # one transaction
        register_vector(conn)
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(INSERT, rows)
    return len(rows)


def set_permissions(doc_id: str, allowed_groups: list[str]) -> None:
    """Sharing changed at the source: update ACLs without re-embedding.
    Call it from a sync job (Feature 04) or a webhook handler."""
    with psycopg.connect(DATABASE_URL) as conn:
        if allowed_groups:
            conn.execute("UPDATE chunks SET allowed_groups = %s"
                         " WHERE doc_id = %s", (allowed_groups, doc_id))
        else:  # unshared with everyone: remove it, don't leave it open
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
