"""Tests for rglr.rangayyan_logistic_regression."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaclass import rangayyan_logistic_regression


def test_rglr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    lr = 0.01
    max_iter = 100
    result = rangayyan_logistic_regression(X, y, lr, max_iter)
    assert isinstance(result, dict)
    assert any(k in result for k in ("estimate", "coefficients", "coef", "params"))


def test_rglr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    lr = 0.001
    max_iter = 50
    result = rangayyan_logistic_regression(X, y, lr, max_iter)
    assert isinstance(result, dict)
    assert any(k in result for k in ("estimate", "coefficients", "coef", "params"))
