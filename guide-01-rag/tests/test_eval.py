"""Chapter 08's retrieval metrics, the eval runner, and Chapter 06's tool."""
import json

import pytest

import run_eval
from eval_retrieval import evaluate, hit_at_k, recall_at_k, reciprocal_rank
from search_tool import SEARCH_TOOL, execute_search


def test_worked_example_from_the_chapter():
    """4 relevant chunks; 5 returned, relevant ones at ranks 1, 3 and 4."""
    retrieved = ["r1", "x", "r2", "r3", "y"]
    relevant = {"r1", "r2", "r3", "r4"}
    assert recall_at_k(retrieved, relevant, 5) == 0.75
    assert hit_at_k(retrieved, relevant, 5) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_evaluate_averages_over_questions():
    ranked = {"q1": ["a", "b"], "q2": ["x", "c"]}
    golden = [{"question": "q1", "relevant": {"a"}},
              {"question": "q2", "relevant": {"c"}}]
    assert evaluate(ranked.get, golden, k=1) == {
        "recall@1": 0.5, "hit_rate@1": 0.5, "mrr": 0.75, "n": 2}


def test_eval_runner_compares_every_configuration(capsys):
    results = run_eval.main(["--k", "1"])
    assert set(results) == {"dense", "bm25", "hybrid (RRF)",
                            "hybrid + rerank"}
    for slices in results.values():
        assert set(slices) == {"ALL", "exact term", "paraphrase"}
        assert slices["ALL"]["n"] == 9
    # BM25 finds exact codes: every exact-term question's chunk is retrieved.
    assert results["bm25"]["exact term"]["recall@1"] == 1.0
    assert "hybrid + rerank" in capsys.readouterr().out


@pytest.fixture(scope="module")
def index():
    return run_eval.load_index("docs")


def test_search_tool_definition_is_valid_json_schema():
    assert json.loads(json.dumps(SEARCH_TOOL)) == SEARCH_TOOL
    assert SEARCH_TOOL["input_schema"]["required"] == ["query"]


@pytest.mark.parametrize("args, error", [
    ({}, "query must be a non-empty string"),
    ({"query": "  "}, "query must be a non-empty string"),
    ({"query": "returns", "k": 0}, "k must be an integer from 1 to 20"),
    ({"query": "returns", "k": "5"}, "k must be an integer from 1 to 20"),
])
def test_search_tool_returns_errors_the_model_can_fix(index, args, error):
    assert execute_search(index, args) == {"ok": False, "error": error}


def test_search_tool_returns_chunks_with_ids(index):
    out = execute_search(index, {"query": "error E-405 locked", "k": 2})
    assert out["ok"] and len(out["data"]) == 2
    assert out["data"][0]["id"] == "errors#0"
    assert set(out["data"][0]) == {"id", "title", "updated", "text", "score"}
