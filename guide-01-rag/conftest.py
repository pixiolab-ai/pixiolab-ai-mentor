"""Shared test setup for guide-01-rag. pytest loads this file first.

rag.py loads two open models from Hugging Face when it is imported. The
tests replace the sentence_transformers module with small deterministic
fakes before anything imports rag, so they need no download, no PyTorch and
no network. The Anthropic client is replaced per test (tests/test_rag.py);
as a safety net, its base URL points at a closed local port.
"""
import hashlib
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"

DIM = 384
PREFIX = "Represent this sentence for searching relevant passages: "


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class FakeSentenceTransformer:
    """Hashed bag of words: texts that share words get similar vectors."""

    def __init__(self, name: str):
        self.name = name

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in words(text.removeprefix(PREFIX)):
            v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1
        return v / (np.linalg.norm(v) or 1.0)

    def encode(self, texts, normalize_embeddings=False):
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(t) for t in texts])


class FakeCrossEncoder:
    """Scores a (query, passage) pair by the share of query words found."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            q, p = set(words(query)), set(words(passage))
            scores.append(len(q & p) / (len(q) or 1))
        return np.array(scores, dtype=np.float32)


fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
fake_module.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_module
