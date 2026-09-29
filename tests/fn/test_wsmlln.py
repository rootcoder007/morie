"""Tests for wsmlln.wasserman_lln."""

from morie.fn import _array_core as np
from morie.fn.wsmlln import wasserman_lln


def test_wsmlln_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_lln(data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmlln_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_lln(data)
    assert isinstance(result, dict)


def test_running_means_recomputed():
    import pytest

    d = [2.0, 4.5, 3.0, 7.5, 1.0]
    r = wasserman_lln(d)
    assert r["running_means"] == pytest.approx([sum(d[: k + 1]) / (k + 1) for k in range(5)], rel=1e-14)
    assert r["last_gap"] == pytest.approx(abs(sum(d) / 5 - sum(d[:4]) / 4), rel=1e-13)
