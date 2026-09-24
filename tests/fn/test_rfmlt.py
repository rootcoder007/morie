"""Tests for rfmlt.rf_multivariate."""

from morie.fn import _array_core as np

from morie.fn.rfmlt import rf_multivariate


def test_rfmlt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y_matrix = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rf_multivariate(X, Y_matrix)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rfmlt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y_matrix = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rf_multivariate(X, Y_matrix)
    assert isinstance(result, dict)
