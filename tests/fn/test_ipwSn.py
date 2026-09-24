"""Tests for ipwSn.ipw_sensitivity."""

import math

from morie.fn import _array_core as np
from morie.fn.ipwSn import ipw_sensitivity


def test_ipwSn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 80
    p = 3
    Y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, p))
    u = rng.uniform(0, 1, n)
    C = [1.0 if x < 0.6 else 0.0 for x in u]
    if sum(C) == 0:
        C[0] = 1.0
    lam_grid = np.linspace(-1.0, 1.0, 11)
    result = ipw_sensitivity(Y, X, C, lam_grid)
    assert hasattr(result, "payload")
    payload = result.payload
    assert "estimate" in payload
    assert math.isfinite(float(payload["estimate"]))
    assert "mu" in payload
    assert "lambda" in payload
    assert len(payload["mu"]) == len(lam_grid)
    assert len(payload["lambda"]) == len(lam_grid)
    assert payload["n"] == n
    assert payload["n_observed"] == int(sum(C))


def test_ipwSn_edge():
    """Test edge cases with X=None (intercept-only selection model)."""
    rng = np.random.default_rng(7)
    n = 60
    Y = rng.normal(0, 1, n)
    u = rng.uniform(0, 1, n)
    C = [1.0 if x < 0.5 else 0.0 for x in u]
    if sum(C) == 0:
        C[0] = 1.0
    lam_grid = np.linspace(-0.5, 0.5, 5)
    result = ipw_sensitivity(Y, None, C, lam_grid)
    assert hasattr(result, "payload")
    payload = result.payload
    assert "estimate" in payload
    assert math.isfinite(float(payload["estimate"]))
    assert len(payload["mu"]) == len(lam_grid)
    assert payload["n"] == n
    assert "propensity" in payload
    assert "gamma" in payload
