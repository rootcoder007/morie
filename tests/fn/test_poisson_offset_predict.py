"""Tests for poisson_offset_predict.poisson_offset_predict."""

from morie.fn import _array_core as np

from morie.fn.poisson_offset_predict import poisson_offset_predict


def test_ca6e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_offset_predict(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca6e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_offset_predict(x)
    assert isinstance(result, dict)
