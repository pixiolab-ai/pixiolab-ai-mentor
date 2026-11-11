"""Run against a throwaway database: DATABASE_URL=... pytest tests/"""
import hashlib

import numpy as np
import psycopg
import pytest

import app.rag.ingest as ingest_mod
import app.rag.search as search_mod
from app.config import DATABASE_URL, EMBED_DIM

CANARY = "CANARY-7731-COMP-BANDS"  # made up; appears in one HR doc only
DOCS = [
    ("handbook", "Office hours and holidays for all staff.", ["all-staff"]),
    ("hr-comp-2026", f"Engineering salary bands for 2026. {CANARY}",
     ["hr"]),
    ("sales-playbook", "Discount rules for enterprise deals.", ["sales"]),
]


def fake_embed(texts: list[str]) -> list[np.ndarray]:
    """Deterministic bag of words: shared words give close vectors."""
    vectors = []
    for text in texts:
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        for word in text.lower().split():
            v[int(hashlib.sha256(word.encode()).hexdigest(), 16)
              % EMBED_DIM] += 1
        vectors.append(v / (np.linalg.norm(v) or 1.0))
    return vectors


@pytest.fixture
def seeded_db(monkeypatch):
    monkeypatch.setattr(ingest_mod, "embed", fake_embed)
    monkeypatch.setattr(search_mod, "embed", fake_embed)
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE chunks")
    for doc_id, text, groups in DOCS:
        ingest_mod.ingest(doc_id, text, groups)
