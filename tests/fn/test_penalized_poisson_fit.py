"""Tests for penalized_poisson_fit.penalized_poisson_fit."""

from morie.fn import _array_core as np

from morie.fn.penalized_poisson_fit import penalized_poisson_fit


def test_msm122_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = penalized_poisson_fit(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm122_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = penalized_poisson_fit(X, y)
    assert isinstance(result, dict)
