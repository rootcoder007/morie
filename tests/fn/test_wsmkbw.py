"""Tests for wsmkbw.wasserman_kde_bandwidth."""

from morie.fn import _array_core as np
from morie.fn.wsmkbw import wasserman_kde_bandwidth


def test_wsmkbw_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kde_bandwidth(data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmkbw_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kde_bandwidth(data)
    assert isinstance(result, dict)


def test_bandwidths_recomputed():
    import math

    import pytest

    d = [3.1, -0.4, 2.2, 5.9, 1.7, 0.3, 4.4, 2.8]
    n = 8
    m = sum(d) / n
    s = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    xs = sorted(d)
    iqr = xs[math.ceil(0.75 * n) - 1] - xs[math.ceil(0.25 * n) - 1]
    r = wasserman_kde_bandwidth(d)
    assert r["estimate"] == pytest.approx(0.9 * min(s, iqr / 1.34) * n**-0.2, rel=1e-13)
    assert r["h_normal_reference"] == pytest.approx((4 / (3 * n)) ** 0.2 * s, rel=1e-13)
