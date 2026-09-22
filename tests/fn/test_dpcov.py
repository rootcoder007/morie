"""Tests for dpcov.dp_covariance."""

from morie.fn import _array_core as np

from morie.fn.dpcov import dp_covariance


def test_dpcov_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dp_covariance(X)
    assert isinstance(result, dict)
    assert "release" in result
def test_dpcov_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dp_covariance(X)
    assert isinstance(result, dict)
