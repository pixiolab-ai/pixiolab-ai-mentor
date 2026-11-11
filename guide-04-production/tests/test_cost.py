"""Feature 02: one usage row per call, exact cost, and a once-a-day budget alert.

The model call is replaced by a function that returns a fixed LLMResult, so
these tests check the bookkeeping, not a provider.
"""
import threading
from decimal import Decimal

import psycopg
import pytest

import app.cost.budget as budget_mod
import app.cost.tracking as tracking_mod
from app.config import DATABASE_URL
from app.cost.pricing import cost_usd
from app.llm.client import LLMResult
from app.llm.text import NoText


def result(**overrides) -> LLMResult:
    fields = dict(text="ok", model="claude-sonnet-5-5", fallback=False,
                  input_tokens=1000, output_tokens=500,
                  cache_read_tokens=2000, cache_write_tokens=100)
    fields.update(overrides)
    return LLMResult(**fields)


def rows(sql: str, args=None) -> list[tuple]:
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute(sql, args).fetchall()


def spend_today(amount: str) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "INSERT INTO llm_usage (user_id, feature, model, input_tokens,"
            " output_tokens, cost_usd) VALUES ('u', 'f', 'm', 1, 1, %s)",
            (Decimal(amount),))


def test_cost_matches_the_hand_calculation():
    """The guide's Step 3 checkpoint, with the illustrative prices:
    (1,000 x 2.00 + 500 x 10.00 + 2,000 x 0.10 + 100 x 2.50) / 1,000,000."""
    row = {"model": "claude-sonnet-5-5", "input_tokens": 1000,
           "output_tokens": 500, "cache_read_tokens": 2000,
           "cache_write_tokens": 100}
    assert cost_usd(row) == Decimal("0.007450")


def test_unknown_model_costs_none_instead_of_raising():
    row = {"model": "not-priced", "input_tokens": 1, "output_tokens": 1,
           "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert cost_usd(row) is None


@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
class TestTracking:
    @pytest.fixture(autouse=True)
    def tracking_on(self, monkeypatch):
        monkeypatch.setattr(tracking_mod, "TRACK_USAGE", True)

    def test_every_call_writes_one_row_with_its_cost(self, monkeypatch):
        monkeypatch.setattr(tracking_mod, "generate", lambda m, **o: result())
        tracking_mod.tracked_generate("u1", "summarize", [])
        assert rows("SELECT user_id, feature, model, cost_usd"
                    " FROM llm_usage") == [
            ("u1", "summarize", "claude-sonnet-5-5", Decimal("0.007450"))]

    def test_fallback_row_keeps_its_own_model_and_null_cost(self, monkeypatch):
        monkeypatch.setattr(tracking_mod, "generate",
                            lambda m, **o: result(model="gpt-test",
                                                  fallback=True))
        tracking_mod.tracked_generate("u1", "summarize", [])
        assert rows("SELECT model, fallback, cost_usd FROM llm_usage") == [
            ("gpt-test", True, None)]

    def test_refusal_is_still_logged(self, monkeypatch):
        def refuse(messages, **options):
            raise NoText("refusal", result(text=""))
        monkeypatch.setattr(tracking_mod, "generate", refuse)
        with pytest.raises(NoText):
            tracking_mod.tracked_generate("u1", "summarize", [])
        assert len(rows("SELECT 1 FROM llm_usage")) == 1


def test_database_outage_never_fails_a_paid_call(monkeypatch, caplog):
    monkeypatch.setattr(tracking_mod, "DATABASE_URL",
                        "postgresql://nobody@127.0.0.1:1/none")
    monkeypatch.setattr(tracking_mod, "TRACK_USAGE", True)
    monkeypatch.setattr(tracking_mod, "generate", lambda m, **o: result())
    assert tracking_mod.tracked_generate("u1", "f", []).text == "ok"
    assert "usage row not stored" in caplog.text  # replayable from the log


@pytest.mark.db
@pytest.mark.usefixtures("clean_db")
class TestBudgetAlert:
    @pytest.fixture(autouse=True)
    def tiny_budget(self, monkeypatch):
        monkeypatch.setattr(budget_mod, "DAILY_BUDGET_USD", Decimal("0.01"))

    def test_alerts_once_per_day(self, monkeypatch):
        sent = []
        monkeypatch.setattr(budget_mod, "slack_post",
                            lambda text, channel: sent.append(text))
        spend_today("0.02")
        assert budget_mod.check_budget() == "alerted"
        assert budget_mod.check_budget() == "already alerted today"
        assert len(sent) == 1 and "200%" in sent[0]

    def test_slack_failure_rolls_back_so_the_next_check_retries(
            self, monkeypatch):
        def slack_down(text, channel):
            raise RuntimeError("slack down")
        monkeypatch.setattr(budget_mod, "slack_post", slack_down)
        spend_today("0.02")
        with pytest.raises(RuntimeError):
            budget_mod.check_budget()
        assert rows("SELECT count(*) FROM alerts_sent") == [(0,)]

    def test_concurrent_checks_send_one_alert(self, monkeypatch):
        sent, start = [], threading.Barrier(5)
        monkeypatch.setattr(budget_mod, "slack_post",
                            lambda text, channel: sent.append(text))
        spend_today("0.02")

        def check():
            start.wait()
            budget_mod.check_budget()

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(sent) == 1

    def test_under_80_percent_is_ok(self, monkeypatch):
        monkeypatch.setattr(budget_mod, "slack_post",
                            lambda text, channel: pytest.fail("no alert"))
        spend_today("0.005")
        assert budget_mod.check_budget() == "ok"
