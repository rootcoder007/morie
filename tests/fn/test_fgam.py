"""Tests for fgam.functional_gam."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.fgam import functional_gam


def test_fgam_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n, T = 40, 10
    X = rng_x.normal(0, 1, (n, T))
    Y = rng_y.normal(0, 1, n)
    result = functional_gam(X, Y, basis=4)
    assert isinstance(result, dict)
    # A fitted functional GAM exposes at least one of its model components.
    assert any(
        k in result
        for k in ("coef", "coefficients", "intercept", "theta0",
                  "surface", "fitted", "yhat")
    )


def test_fgam_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n, T = 40, 10
    X = rng_x.normal(0, 1, (n, T))
    Y = rng_y.normal(0, 1, n)
    # Edge case: minimum valid marginal basis sizes (4 each).
    result = functional_gam(X, Y, n_x=4, n_t=4, lam_x=1.0, lam_t=1.0)
    assert isinstance(result, dict)
    assert any(
        k in result
        for k in ("coef", "coefficients", "intercept", "theta0",
                  "surface", "fitted", "yhat")
    )
