"""Tests for raoscot.rao_scott_chisq."""

import pytest

from morie.fn.raoscot import rao_scott_chisq


P_HAT = [0.18, 0.32, 0.27, 0.23]
P0 = [0.25, 0.25, 0.25, 0.25]


def test_raoscot_basic():
    """X^2 = n sum (p - p0)^2 / p0; with V = c P0 every generalised deff
    is c (Rao & Scott 1981, Sec. 2.3), so the correction divides by c."""
    n, c = 400, 1.8
    x2 = n * sum((a - b) ** 2 / b for a, b in zip(P_HAT, P0))
    V = [[c * (P0[i] * (i == j) - P0[i] * P0[j]) for j in range(4)] for i in range(4)]
    r = rao_scott_chisq(P_HAT, P0, n, V=V)
    assert r["statistic"] == pytest.approx(x2, rel=1e-12)
    assert r["lambda_bar"] == pytest.approx(c, rel=1e-12)
    assert r["corrected"] == pytest.approx(x2 / c, rel=1e-12)
    assert r["df"] == 3


def test_raoscot_edge():
    """Equal cell deffs d give lambda_bar = d; none given is the SRS test."""
    r = rao_scott_chisq(P_HAT, P0, 400, deffs=[2.5] * 4)
    assert r["lambda_bar"] == pytest.approx(2.5, rel=1e-12)
    srs = rao_scott_chisq(P_HAT, P0, 400)
    assert srs["lambda_bar"] == 1.0 and srs["corrected"] == srs["statistic"]
    with pytest.raises(ValueError, match="sum to 1"):
        rao_scott_chisq([0.5, 0.6], [0.5, 0.5], 100)
    with pytest.raises(ValueError, match="positive"):
        rao_scott_chisq([0.5, 0.5], [1.0, 0.0], 100)


