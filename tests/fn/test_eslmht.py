"""Tests for eslmht.esl_holm_bonferroni (Holm 1979 step-down).

Anchored on a hand-computed example.  With p sorted .005 .01 .03 .04 .2
and m = 5, alpha = 0.05:

    .005 <= .05/5 = .0100  reject
    .010 <= .05/4 = .0125  reject
    .030 >  .05/3 = .0167  STOP        -> 2 rejections

and the monotone adjusted p-values are .025 .04 .09 .09 .20.
"""

import pytest

from morie.fn.eslmht import esl_holm_bonferroni


def test_holm_hand_computed():
    res = esl_holm_bonferroni([0.005, 0.01, 0.03, 0.04, 0.2], 0.05)
    assert res["n_reject"] == 2
    assert list(res["reject"]) == [True, True, False, False, False]
    assert list(res["p_adjusted"]) == pytest.approx([0.025, 0.04, 0.09, 0.09, 0.20])


def test_holm_is_permutation_invariant():
    a = esl_holm_bonferroni([0.005, 0.01, 0.03, 0.04, 0.2], 0.05)
    b = esl_holm_bonferroni([0.2, 0.04, 0.01, 0.03, 0.005], 0.05)
    assert b["n_reject"] == a["n_reject"]
    assert list(b["reject"]) == [False, False, True, False, True]
    assert list(b["p_adjusted"]) == pytest.approx([0.20, 0.09, 0.04, 0.09, 0.025])


def test_holm_stops_at_first_failure():
    # .001 passes, .049 fails at alpha/2, so the tiny .0011 behind it is
    # NOT rejected even though .0011 <= alpha/1.  That is the whole point
    # of a step-down procedure.
    res = esl_holm_bonferroni([0.001, 0.049, 0.0011], 0.05)
    assert res["n_reject"] == 1
    assert list(res["reject"]) == [True, False, False]


def test_holm_single_hypothesis_is_unadjusted():
    res = esl_holm_bonferroni([0.03], 0.05)
    assert res["n_reject"] == 1
    assert res["p_adjusted"][0] == pytest.approx(0.03)
