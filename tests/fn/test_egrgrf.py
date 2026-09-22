"""Tests for egrgrf.egregious_loss_forest."""

from morie.fn import _array_core as np

from morie.fn.egrgrf import egregious_loss_forest


def test_egrgrf_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = egregious_loss_forest(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_egrgrf_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = egregious_loss_forest(y, D, X)
    assert isinstance(result, dict)
