"""Chapters 08 and 09: retrieval metrics and agent trajectory checks."""
from math import log2

import pytest

from agent_checks import pass_at_k, pass_hat_k, trajectory_violations
from retrieval_metrics import (hit_at_k, ndcg_at_k, precision_at_k,
                               recall_at_k, reciprocal_rank)


def test_retrieval_worked_example():
    """4 relevant chunks; 5 returned, relevant ones at ranks 1, 3 and 4."""
    retrieved = ["a", "x", "b", "c", "y"]
    relevant = {"a", "b", "c", "d"}
    assert recall_at_k(retrieved, relevant, 5) == 0.75
    assert precision_at_k(retrieved, relevant, 5) == 0.6
    assert reciprocal_rank(retrieved, relevant) == 1.0
    assert hit_at_k(retrieved, relevant, 1) == 1.0


def test_reciprocal_rank_of_a_miss_is_zero():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_rewards_putting_the_best_chunk_first():
    grades = {"best": 3, "ok": 1}
    assert ndcg_at_k(["best", "ok"], grades, 2) == 1.0
    swapped = (1 + 3 / log2(3)) / (3 + 1 / log2(3))
    assert ndcg_at_k(["ok", "best"], grades, 2) == pytest.approx(swapped)
    assert ndcg_at_k(["x"], {}, 1) == 0.0


def span(name, **extra):
    return {"kind": "tool", "name": name, **extra}


def test_trajectory_rules_catch_unsafe_paths():
    trace = {"spans": [
        {"kind": "llm"},
        span("lookup_order", args={}),
        span("issue_refund", args={"amount": 250}),   # over the limit
        span("delete_account", args={})]}             # never confirmed
    assert trajectory_violations(trace) == [
        "refund_over_limit_unapproved", "delete_without_confirmation"]


def test_approval_and_lookup_make_a_large_refund_legal():
    trace = {"spans": [
        span("lookup_order", args={}),
        span("request_approval", args={}, result={"approved": True}),
        span("issue_refund", args={"amount": 250})]}
    assert trajectory_violations(trace) == []


def test_refund_before_lookup_and_search_loops():
    trace = {"spans": [span("search_kb", args={})] * 6
             + [span("issue_refund", args={"amount": 20})]}
    assert trajectory_violations(trace) == ["too_many_searches",
                                            "refund_before_lookup"]


def test_pass_at_k_vs_pass_hat_k():
    """A task that passed 8 of 10 trials looks perfect by pass@4, but works
    four times in a row only about one time in three (pass^4)."""
    assert pass_at_k(10, 8, 4) == 1.0
    assert pass_hat_k(10, 8, 4) == pytest.approx(1 / 3)
