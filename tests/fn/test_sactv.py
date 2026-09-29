"""Tests for morie.fn.sactv: values recomputed from the definition."""

from morie.fn.sactv import activity


def test_hjorth_activity_is_the_population_variance():
    x = [1.0, 2.0, 6.0, -3.0, 0.5]
    m = sum(x) / 5
    assert abs(activity(x).value - sum((v - m) ** 2 for v in x) / 5) < 1e-14
