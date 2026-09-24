"""Tests for josmap.joseph_smape."""

from morie.fn import _array_core as np

from morie.fn.josmap import joseph_smape


def test_josmap_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_smape(y, yhat)
    assert isinstance(result, dict)
    assert "smape" in result


def test_josmap_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_smape(y, yhat)
    assert isinstance(result, dict)
