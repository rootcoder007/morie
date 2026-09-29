"""Tests for morie.fn.bayes_simple: values recomputed from first principles."""

from morie.fn.bayes_simple import bayes_simple


def test_bayes_formula():
    assert abs(bayes_simple(0.9, 0.01, 0.05)["posterior"] - 0.9 * 0.01 / 0.05) < 1e-16
    assert abs(bayes_simple(0.3, 0.5, 0.6)["posterior"] - 0.25) < 1e-15
