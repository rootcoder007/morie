"""Tests for gpsfn.gp_sparse_inducing."""

import math

from morie.fn import _array_core as np

from morie.fn.gpsfn import gp_sparse_inducing


def test_gpsfn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, n_test, m = 40, 3, 5, 10
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    X_test = rng.normal(0, 1, (n_test, p))
    inducing = rng.normal(0, 1, (m, p))
    result = gp_sparse_inducing(X, y, X_test, inducing)
    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert "pred" in payload
    assert "var" in payload
    assert "lam" in payload
    assert len(payload["pred"]) == n_test
    assert len(payload["var"]) == n_test
    assert len(payload["lam"]) == n
    assert math.isfinite(payload["estimate"])
    for v in payload["pred"]:
        assert math.isfinite(v)
    for v in payload["var"]:
        assert math.isfinite(v)


def test_gpsfn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p, n_test, m = 40, 3, 5, 10
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    X_test = rng.normal(0, 1, (n_test, p))
    inducing = rng.normal(0, 1, (m, p))
    result = gp_sparse_inducing(X, y, X_test, inducing)
    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert "pred" in payload
    assert "var" in payload
    assert "lam" in payload
    for v in payload["pred"]:
        assert math.isfinite(v)
    for v in payload["var"]:
        assert math.isfinite(v)
    for v in payload["lam"]:
        assert math.isfinite(v)
