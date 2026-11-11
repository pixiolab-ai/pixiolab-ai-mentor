from functools import cache

import numpy as np
from openai import OpenAI

from app.config import EMBED_DIM, EMBED_MODEL


@cache
def _client() -> OpenAI:
    return OpenAI()


def embed(texts: list[str]) -> list[np.ndarray]:
    """Any provider works if the size matches vector(N). Drop
    `dimensions` if your model has a fixed size."""
    resp = _client().embeddings.create(
        model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def split_into_chunks(text: str, size: int = 1500) -> list[str]:
    """Pack paragraphs into ~size-character chunks (Guide 01: better)."""
    chunks, current = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if current and len(current) + len(para) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    return [c for c in chunks + [current] if c]
