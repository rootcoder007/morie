"""Tests for evgpdl.evt_gpd_loglik."""

import numpy as np

from morie.fn.evgpdl import evt_gpd_loglik


def test_evgpdl_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_xi = np.random.default_rng(42)
    y = np.abs(rng_y.normal(0, 1, 100)) + 0.1
    sigma = 1.0
    xi = float(rng_xi.normal(0, 0.1))
    result = evt_gpd_loglik(y, sigma, xi)
    assert isinstance(result, dict)
    assert "ll" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == len(y)

    # Independent recomputation of the GPD log-likelihood (Coles 2001 eq. 4.10).
    s = float(sigma)
    x = float(xi)
    arr = np.asarray(y, dtype=float)
    if abs(x) < 1e-12:
        expected_ll = float(-len(arr) * np.log(s) - (1.0 / s) * np.sum(arr - 0.0))
    else:
        z = arr / s
        w = 1.0 + x * z
        if np.any(w <= 0):
            expected_ll = float("-inf")
        else:
            expected_ll = float(
                -len(arr) * np.log(s)
                - (1.0 + 1.0 / x) * np.sum(np.log(w))
            )
    assert result["ll"] == expected_ll


def test_evgpdl_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_xi = np.random.default_rng(42)
    y = np.abs(rng_y.normal(0, 1, 100)) + 0.1
    sigma = 1.0
    xi = float(rng_xi.normal(0, 0.1))
    result = evt_gpd_loglik(y, sigma, xi)
    assert isinstance(result, dict)
    assert "ll" in result
    assert "n" in result
    assert result["n"] == len(y)
