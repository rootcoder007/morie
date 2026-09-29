"""Tests for morie.fn.prob_or_general: values recomputed from first principles."""

from morie.fn.prob_or_general import prob_or_general


def test_general_or_rule():
    r = prob_or_general(0.5, 0.4, 0.2)
    assert abs(r["p_or"] - 0.7) < 1e-15
