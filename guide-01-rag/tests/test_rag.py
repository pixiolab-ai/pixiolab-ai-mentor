"""Chapter 04's pipeline: chunk, index, hybrid search, rerank, cited answer.

The embedding and reranking models are deterministic fakes (../conftest.py)
and the LLM is a scripted fake client, so these tests check the pipeline's
logic, not model quality. run_eval.py measures retrieval with real models.
"""
import runpy
import sys
from types import SimpleNamespace as NS

import anthropic
import pytest

import rag


def paragraph(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.fixture(scope="module")
def index():
    from run_eval import load_index
    return load_index("docs")


# ---------- Part 1: chunk ----------

def test_chunks_pack_whole_paragraphs_and_carry_overlap():
    text = "\n\n".join([paragraph("a", 100), paragraph("b", 100),
                        paragraph("c", 100)])
    chunks = rag.chunk_doc("guide", text, max_words=220, overlap=40)
    assert [c.id for c in chunks] == ["guide#0", "guide#1"]
    first, second = (c.text.split() for c in chunks)
    assert len(first) == 200                 # a and b fit; c would not
    assert second[:40] == first[-40:]        # the overlap carried forward
    assert second[40:] == paragraph("c", 100).split()


def test_a_short_document_is_one_chunk():
    (chunk,) = rag.chunk_doc("returns", "One line.\n\nAnother line.")
    assert chunk == rag.Chunk("returns#0", "returns", "One line. Another line.")


def test_tokenizer_keeps_error_codes_whole():
    assert rag.tokenize("Error E-404 on login!") == ["error", "e-404", "on",
                                                    "login"]


# ---------- Part 2: hybrid search, RRF, rerank ----------

def test_rrf_rewards_agreement_between_rankings():
    # 1 is second and first; 0 is first and third: 1/62 + 1/61 > 1/61 + 1/63
    assert rag.rrf([[0, 1, 2], [1, 3, 0]]) == [1, 0, 3, 2]


def test_keyword_search_drops_documents_with_no_matching_terms(index):
    hits = [index.chunks[i].id for i in index.keyword("e-404", 50)]
    assert hits == ["errors#0"]


def test_search_returns_reranked_chunks_best_first(index):
    results = rag.search(index, "error E-404 on login", k_final=3)
    assert results[0][0].id == "errors#0"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) for s in scores) and len(results) <= 3


# ---------- Part 3: generate with citations, then check them ----------

def fake_client(answer_text: str, requests: list):
    def create(**request):
        requests.append(request)
        return NS(stop_reason="end_turn", content=[
            NS(type="thinking", thinking="", signature="s"),
            NS(type="text", text=answer_text)])
    return NS(messages=NS(create=create))


def test_answer_tags_sources_and_flags_citations_it_never_sent(
        index, monkeypatch):
    requests = []
    monkeypatch.setattr(rag, "client", fake_client(
        "Yes, within 60 days with a receipt [returns#0]. Boots too "
        "[warranty#2].", requests))
    text, cited, unknown = rag.answer(index, "Can I return boots after 40 days?")
    assert text.startswith("Yes, within 60 days")  # thinking block skipped
    assert cited == ["returns#0", "warranty#2"]
    assert unknown == ["warranty#2"]               # a hallucinated citation
    request = requests[0]
    assert request["system"] == rag.SYSTEM
    prompt = request["messages"][0]["content"]
    assert '<source id="returns#0">' in prompt
    assert prompt.endswith("Question: Can I return boots after 40 days?")


def test_command_line(monkeypatch, capsys):
    requests = []
    fake = fake_client("Yes [returns#0].", requests)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)  # for the rerun
    monkeypatch.setattr(sys, "argv", ["rag.py", "docs", "Can I return boots?"])
    runpy.run_path("rag.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "cited: ['returns#0'] | not in sources: []" in out
