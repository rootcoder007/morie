"""Tests for sampvar (Morin eq 3.73)."""

import pytest

from morie.fn.sampvar import sampvar


def test_sampvar_both_divisors():
    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    ss = sum((v - m) ** 2 for v in x)
    r = sampvar(x)
    assert r["sample_variance"] == pytest.approx(ss / 4, rel=1e-14)
    assert r["population_variance"] == pytest.approx(ss / 5, rel=1e-14)
