"""9.03 chunker with overlap, 9.04 cosine top-k, 9.05 reciprocal rank fusion."""
import numpy as np
import pytest

from chunker import chunk
from rrf import rrf
from topk import top_k_cosine

TOKENS = [str(i) for i in range(10)]


# ---------- 9.03 chunker ----------

def texts(chunks):
    return [c["text"] for c in chunks]


def test_neighbors_share_the_overlap():
    assert texts(chunk(TOKENS, size=4, overlap=1)) == [
        ["0", "1", "2", "3"], ["3", "4", "5", "6"], ["6", "7", "8", "9"]]


def test_no_tiny_duplicate_tail():
    assert texts(chunk(TOKENS[:7], size=4, overlap=1)) == [
        ["0", "1", "2", "3"], ["3", "4", "5", "6"]]


def test_offsets_point_back_into_the_source():
    last = chunk(TOKENS, size=4, overlap=0)[-1]
    assert (last["start"], last["end"], last["text"]) == (8, 10, ["8", "9"])


def test_edge_cases():
    assert chunk([], 4, 1) == []
    assert texts(chunk(TOKENS[:3], 4, 1)) == [["0", "1", "2"]]


@pytest.mark.parametrize("size, overlap", [(0, 0), (4, 4), (4, -1)])
def test_settings_that_would_never_advance_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk(TOKENS, size, overlap)


def test_the_follow_up_question_10k_tokens_gives_30_chunks():
    chunks = chunk([str(i) for i in range(10_000)], size=400, overlap=60)
    assert len(chunks) == 30
    assert (chunks[-1]["start"], chunks[-1]["end"]) == (9860, 10_000)


# ---------- 9.04 cosine top-k ----------

@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    return rng.normal(size=(1000, 64)), rng.normal(size=64)


def test_matches_a_full_sort(corpus):
    docs, query = corpus
    ids, scores = top_k_cosine(query, docs, k=5)
    reference = (docs / np.linalg.norm(docs, axis=1, keepdims=True)) @ (
        query / np.linalg.norm(query))
    assert list(ids) == list(np.argsort(-reference)[:5])
    assert np.allclose(scores, reference[ids])


def test_k_is_clamped_to_the_corpus(corpus):
    docs, query = corpus
    assert len(top_k_cosine(query, docs[:3], k=5)[0]) == 3
    assert len(top_k_cosine(query, docs[:0], k=5)[0]) == 0
    assert len(top_k_cosine(query, docs, k=0)[0]) == 0


def test_a_zero_vector_does_not_divide_by_zero(corpus):
    docs, _ = corpus
    ids, scores = top_k_cosine(np.zeros(64), docs, k=3)
    assert len(ids) == 3 and np.all(np.isfinite(scores))


# ---------- 9.05 RRF ----------

def test_the_worked_example():
    fused = rrf([["A", "B", "C"], ["B", "D", "A"]])
    assert [doc for doc, _ in fused] == ["B", "A", "D", "C"]
    assert round(fused[0][1], 5) == 0.03252   # 1/62 + 1/61
    assert round(fused[1][1], 5) == 0.03227   # 1/61 + 1/63


def test_a_duplicate_in_one_list_counts_once():
    fused = dict(rrf([["A", "A", "B"]]))
    assert fused == {"A": 1 / 61, "B": 1 / 63}


def test_ties_break_by_id_and_top_n_cuts():
    fused = rrf([["B"], ["A"]], top_n=1)
    assert fused == [("A", 1 / 61)]
