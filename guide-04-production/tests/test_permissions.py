import app.rag.answer as answer_mod
from app.llm.client import LLMResult
from app.rag.search import search

CANARY = "CANARY-7731-COMP-BANDS"  # planted in hr-comp-2026 (conftest.py)

SALES, HR = ["sales", "all-staff"], ["hr", "all-staff"]


def test_sales_cannot_retrieve_hr_docs(seeded_db):
    hits = search("engineering salary bands", SALES)
    assert all(doc_id != "hr-comp-2026" for doc_id, _ in hits)
    assert all(CANARY not in content for _, content in hits)


def test_hr_can_retrieve_hr_docs(seeded_db):
    hits = search("engineering salary bands", HR)
    assert hits[0][0] == "hr-comp-2026"


def test_no_groups_sees_nothing(seeded_db):
    assert search("anything at all", []) == []


def echo_model(user_id, feature, messages, **options):
    """Stands in for the model and answers with its entire prompt. If the
    canary is not in the answer, it was never in the prompt."""
    return LLMResult(text=messages[0]["content"], model="echo",
                     fallback=False, input_tokens=0, output_tokens=0)


def test_answer_path_never_sees_the_canary(seeded_db, monkeypatch):
    monkeypatch.setattr(answer_mod, "tracked_generate", echo_model)
    out = answer_mod.handle_ask(
        "u-sales", {"question": "What are the salary bands?",
                    "groups": SALES})
    assert CANARY not in out["answer"]
    assert "hr-comp-2026" not in out["sources"]
