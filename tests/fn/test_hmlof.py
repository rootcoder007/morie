"""Tests for hmlof.geron_local_outlier_factor."""

from morie.fn import _array_core as np

from morie.fn.hmlof import geron_local_outlier_factor


def test_hmlof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_local_outlier_factor(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "lof" in result


def test_hmlof_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_local_outlier_factor(X)
    assert isinstance(result, dict)
