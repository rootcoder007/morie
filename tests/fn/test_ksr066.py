"""Tests for ksr066.kosorok_ch3_z_estimator_no_bias."""

import pytest

from morie.fn.ksr066 import kosorok_ch3_z_estimator_no_bias


def test_ksr066_basic():
    """An efficient score whose bias is second order in theta - theta0
    satisfies the no-bias condition: bias / (n^-1/2 + |theta - theta0|)
    shrinks along the sequence."""
    ns = [100, 400, 1600, 6400]
    th = [0.5 + n ** -0.5 for n in ns]
    r = kosorok_ch3_z_estimator_no_bias(lambda t, n: (t - 0.5) ** 2, th, 0.5, ns)
    assert r["holds"] is True
    for b, t_, n in zip(r["bias"].tolist(), th, ns):
        assert b == pytest.approx((t_ - 0.5) ** 2, rel=1e-12)


def test_ksr066_edge():
    """A constant first-order bias fails it."""
    ns = [100, 400, 1600, 6400]
    th = [0.5 + n ** -0.5 for n in ns]
    assert kosorok_ch3_z_estimator_no_bias(lambda t, n: 0.1, th, 0.5, ns)["holds"] is False
    with pytest.raises(ValueError, match="entries"):
        kosorok_ch3_z_estimator_no_bias(lambda t, n: 0.0, th[:2], 0.5, ns)


