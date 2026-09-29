"""Tests for morie.fn.dcsub: values recomputed from the definition."""

from morie.fn.dcsub import dc_removal


def test_mean_removed():
    x = [1.0, 2.0, 6.0, -3.0]
    r = dc_removal(x)
    m = sum(x) / 4
    assert max(abs(a - (b - m)) for a, b in zip(r.filtered, x)) < 1e-15
    assert abs(r.extra["dc_value"] - m) < 1e-15
