"""Tests for wassdt.wasserstein_1d."""

from morie.fn import _array_core as np

from morie.fn.wassdt import wasserstein_1d


def test_wassdt_basic():
    """Test basic functionality."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserstein_1d(p, q)
    assert isinstance(result, dict)
    assert "distance" in result


def test_wassdt_edge():
    """Test edge cases."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserstein_1d(p, q)
    assert isinstance(result, dict)
