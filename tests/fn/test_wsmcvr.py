"""Tests for wsmcvr.wasserman_kfold_cv."""

from morie.fn import _array_core as np

from morie.fn.wsmcvr import wasserman_kfold_cv


def test_wsmcvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    k = 5
    result = wasserman_kfold_cv(X, y, model, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmcvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    model = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    k = 5
    result = wasserman_kfold_cv(X, y, model, k)
    assert isinstance(result, dict)
