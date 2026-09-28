"""Tests for evgpdl.evt_gpd_loglik."""

import math

from morie.fn import _array_core as np
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

    # Independent recomputation of the GPD log-likelihood (Coles 2001 eq. 4.10),
    # one term per excess, summed exactly.
    s = float(sigma)
    x = float(xi)
    ys = [float(v) for v in y.tolist()]
    if abs(x) < 1e-8:
        terms = [-math.log(s) - v / s for v in ys]
    else:
        terms = [-math.log(s) - (1.0 + 1.0 / x) * math.log(1.0 + x * v / s)
                 if 1.0 + x * v / s > 0 else float("-inf") for v in ys]
    expected_ll = math.fsum(terms)
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
