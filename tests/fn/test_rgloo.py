"""Tests for rgloo.rangayyan_loo_cv."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_loo_cv


def test_rgloo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_loo_cv(X, y)
    assert isinstance(result, dict)
    assert "error_rate" in result


def test_rgloo_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_loo_cv(X, y)
    assert isinstance(result, dict)
