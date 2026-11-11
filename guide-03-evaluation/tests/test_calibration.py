"""Chapter 07: judge calibration, checked against the chapter's worked example.

200 human-labeled answers: 160 PASS, 40 FAIL. The judge passes 150 of the
160 good ones and fails 26 of the 40 bad ones.
"""
import pytest

from calibration import calibration, corrected_pass_rate, wilson

HUMAN = [True] * 160 + [False] * 40
JUDGE = [True] * 150 + [False] * 10 + [False] * 26 + [True] * 14


def test_tpr_tnr_and_kappa():
    stats = calibration(HUMAN, JUDGE)
    assert stats["tpr"] == pytest.approx(0.9375)   # 150 / 160
    assert stats["tnr"] == pytest.approx(0.65)     # 26 / 40
    assert stats["agreement"] == pytest.approx(0.88)
    assert stats["kappa"] == pytest.approx(0.61, abs=0.005)


def test_raw_agreement_flatters_a_judge_that_passes_everything():
    stats = calibration(HUMAN, [True] * 200)
    assert stats["agreement"] == pytest.approx(0.80)  # looks fine...
    assert stats["tnr"] == 0 and stats["kappa"] == pytest.approx(0.0)


def test_wilson_interval_for_the_tnr():
    low, high = wilson(26, 40)
    assert (low, high) == pytest.approx((0.495, 0.779), abs=0.001)


def test_corrected_pass_rate():
    stats = calibration(HUMAN, JUDGE)
    assert corrected_pass_rate(0.78, stats["tpr"], stats["tnr"]) == \
        pytest.approx(0.732, abs=0.001)


def test_a_judge_no_better_than_chance_cannot_correct_anything():
    with pytest.raises(ValueError):
        corrected_pass_rate(0.5, tpr=0.5, tnr=0.5)
