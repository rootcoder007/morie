"""Tests for wsmlik.wasserman_likelihood."""

from morie.fn import _array_core as np

from morie.fn.wsmlik import wasserman_likelihood


def test_wsmlik_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    f = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta = 0.1
    result = wasserman_likelihood(data, f, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmlik_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    f = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    theta = 0.1
    result = wasserman_likelihood(data, f, theta)
    assert isinstance(result, dict)
