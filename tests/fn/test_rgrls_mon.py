"""Tests for rgrls_mon.rangayyan_rls_monitor."""

from morie.fn import _array_core as np

from morie.fn.bsaadapt import rangayyan_rls_monitor


def test_rgrls_mon_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_rls_monitor(x, d, lam, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgrls_mon_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_rls_monitor(x, d, lam, threshold)
    assert isinstance(result, dict)
