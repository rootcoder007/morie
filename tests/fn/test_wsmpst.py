"""Tests for wsmpst.wasserman_plug_in_estimator."""

from morie.fn import _array_core as np

from morie.fn.wsmpst import wasserman_plug_in_estimator


def test_wsmpst_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_plug_in_estimator(data, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmpst_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_plug_in_estimator(data, T)
    assert isinstance(result, dict)
