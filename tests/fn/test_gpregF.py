"""Tests for gpregF.gp_regression."""

from morie.fn import _array_core as np

from morie.fn.gpregF import gp_regression


def test_gpregF_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (10, 2))
    y = rng.normal(0, 1, 10)
    X_star = rng.normal(0, 1, (5, 2))
    kernel = (1.0, 1.0)
    result = gp_regression(X, y, X_star, kernel)
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    assert "estimate" in result
    assert "weights" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 10
    assert len(result["estimate"]) == 5
    assert len(result["weights"]) == 10


def test_gpregF_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (8, 3))
    y = rng.normal(0, 1, 8)
    X_star = rng.normal(0, 1, (3, 3))
    kernel = (2.0, 0.5)
    result = gp_regression(X, y, X_star, kernel, noise=0.1)
    assert isinstance(result, dict) or hasattr(result, "__getitem__")
    assert "estimate" in result
    assert "weights" in result
    assert result["n"] == 8
    assert len(result["estimate"]) == 3
    assert len(result["weights"]) == 8
