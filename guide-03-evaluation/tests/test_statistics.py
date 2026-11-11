"""Chapter 04: the paired tests, checked against the chapter's worked example.

Two runs on the same 500 cases both gain 2 points (a net 10 cases). In run A
20 cases flip; in run B 40 do. Only run A's gain is distinguishable from noise.
"""
import pytest

from stats import mcnemar_exact, paired_bootstrap


def runs(got_worse: int, got_better: int, n=500, both_pass=390):
    old = [True] * both_pass + [True] * got_worse + [False] * got_better
    new = [True] * both_pass + [False] * got_worse + [True] * got_better
    rest = n - len(old)
    return old + [False] * rest, new + [False] * rest


def test_run_a_is_probably_a_real_gain():
    old, new = runs(got_worse=5, got_better=15)
    b, c, p = mcnemar_exact(old, new)
    assert (b, c) == (5, 15) and p == pytest.approx(0.041, abs=0.001)
    diff, lo, hi = paired_bootstrap(old, new)
    assert diff == pytest.approx(0.02)
    assert (lo, hi) == pytest.approx((0.004, 0.038))  # +0.4 to +3.8 points


def test_run_b_cannot_be_told_from_noise():
    old, new = runs(got_worse=15, got_better=25)
    _, _, p = mcnemar_exact(old, new)
    assert p == pytest.approx(0.154, abs=0.001)
    _, lo, hi = paired_bootstrap(old, new)
    assert lo < 0 < hi
    assert (lo, hi) == pytest.approx((-0.004, 0.044))


def test_no_flips_means_no_evidence_either_way():
    same = [True, False, True]
    assert mcnemar_exact(same, same) == (0, 0, 1.0)


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    old, new = runs(5, 15, n=100, both_pass=60)
    assert paired_bootstrap(old, new, iters=500) == paired_bootstrap(
        old, new, iters=500)
