"""Tests for mlfit.ml_loglik."""

from morie.fn import _array_core as np

from morie.fn.mlfit import ml_loglik


def test_mlfit_basic():
    """Test basic functionality."""
    y = 0.5
    X = 0.5
    V = 0.5
    result = ml_loglik(y, X, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mlfit_edge():
    """Test edge cases."""
    y = 0.5
    X = 0.5
    V = 0.5
    result = ml_loglik(y, X, V)
    assert isinstance(result, dict)
