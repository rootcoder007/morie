"""Tests for wsmifn.wasserman_influence_function."""

from morie.fn import _array_core as np

from morie.fn.wsmifn import wasserman_influence_function


def test_wsmifn_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_influence_function(data, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmifn_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_influence_function(data, T)
    assert isinstance(result, dict)
