"""Tests for gb_rnkci.gibbons_rank_ci."""

from morie.fn import _array_core as np
from morie.fn.gb_rnkci import gibbons_rank_ci


def test_gb_rnkci_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rank_ci(x, alpha)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb_rnkci_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rank_ci(x, alpha)
    assert isinstance(result, dict)


def test_rank_interval_order_statistics():
    v = [4.0, 1.5, 3.2, 8.1, 2.2, 6.6, 5.0]
    s = sorted(v)
    r = gibbons_rank_ci(v, 1, level=0.9)
    assert (r["lower"], r["upper"]) == (s[1], s[5])
    assert r["estimate"] == s[3] and r["level"] == 0.9
    assert gibbons_rank_ci([1.0, 4.0, 2.0, 3.0], 0)["estimate"] == 2.5
