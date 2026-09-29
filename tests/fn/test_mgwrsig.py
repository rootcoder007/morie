"""Tests for morie.fn.mgwrsig."""

from morie.fn.mgwrsig import mgwrsig

E = [0.3, -0.7, 0.25, 0.1, -0.45, 0.6, -0.05]


def test_sigma2():
    assert abs(mgwrsig(E, 3.2).statistic - sum(v * v for v in E) / (7 - 3.2)) < 1e-15
    assert abs(mgwrsig(E, 3.2, n=10).statistic - sum(v * v for v in E) / 6.8) < 1e-15
