"""Tests for hrzp2.horowitz_plr_bandwidth."""

from morie.fn import _array_core as np
from morie.fn.hrzp2 import horowitz_plr_bandwidth


def test_hrzp2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_plr_bandwidth(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hrzp2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_plr_bandwidth(x, y)
    assert isinstance(result, dict)


def test_silverman_plr_bandwidth_recomputed():
    import math

    import pytest

    z = [3.1, -0.4, 2.2, 5.9, 1.7, 0.3, 4.4, 2.8]
    n = 8
    m = sum(z) / n
    s = math.sqrt(sum((v - m) ** 2 for v in z) / (n - 1))
    xs = sorted(z)

    def q(p):
        h = (n - 1) * p
        lo = int(h)
        return xs[lo] + (h - lo) * (xs[lo + 1] - xs[lo])

    sig = min(s, (q(0.75) - q(0.25)) / 1.349)
    r = horowitz_plr_bandwidth(z, z, c=0.9)
    assert r["estimate"] == pytest.approx(0.9 * sig * n**-0.2, rel=1e-12)
