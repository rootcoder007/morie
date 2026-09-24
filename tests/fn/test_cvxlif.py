"""Tests for cvxlif.boyd_linf_fitting."""

import math

from morie.fn import _array_core as np

from morie.fn.cvxlif import boyd_linf_fitting


def test_cvxlif_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    A = rng.normal(0, 1, (n, p))
    b = rng.normal(0, 1, n)
    result = boyd_linf_fitting(A, b)
    assert isinstance(result, dict)
    for key in ("x", "residual", "linf_norm", "n_active", "equioscillates"):
        assert key in result
    x = np.asarray(result["x"])
    assert x.shape == (p,)
    resid = np.asarray(result["residual"])
    assert resid.shape == (n,)
    linf = float(result["linf_norm"])
    assert math.isfinite(linf)
    assert linf >= 0.0
    assert isinstance(result["n_active"], int)
    assert result["n_active"] >= 0
    assert isinstance(result["equioscillates"], bool)


def test_cvxlif_edge():
    """Test edge cases with a small well-determined problem."""
    rng = np.random.default_rng(7)
    n, p = 10, 2
    A = rng.normal(0, 1, (n, p))
    b = rng.normal(0, 1, n)
    result = boyd_linf_fitting(A, b)
    assert isinstance(result, dict)
    linf = float(result["linf_norm"])
    assert math.isfinite(linf)
    assert linf >= 0.0
    assert isinstance(result["n_active"], int)
    assert result["n_active"] >= 0
    assert isinstance(result["equioscillates"], bool)
    x = np.asarray(result["x"])
    assert x.shape == (p,)
    resid = np.asarray(result["residual"])
    assert resid.shape == (n,)
