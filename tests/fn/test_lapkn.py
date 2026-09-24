"""Tests for lapkn.laplacian_kernel."""

from morie.fn import _array_core as np

from morie.fn.lapkn import laplacian_kernel


def test_lapkn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = laplacian_kernel(X)
    assert isinstance(result, dict)
    assert "K" in result


def test_lapkn_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = laplacian_kernel(X)
    assert isinstance(result, dict)
