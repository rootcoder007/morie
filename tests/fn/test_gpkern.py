"""Tests for gpkern.gp_kernel_compose."""

import math

from morie.fn import _array_core as np
from morie.fn.gpkern import gp_kernel_compose


def test_gpkern_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, d = 40, 3
    X = rng.normal(0, 1, (n, d))
    Y = rng.normal(0, 1, (n, d))
    kernel_spec = {
        "op": "sum",
        "parts": [
            {"type": "rbf", "lengthscale": 1.0, "variance": 1.0},
            {"type": "rbf", "lengthscale": 0.5, "variance": 0.3},
        ],
    }
    result = gp_kernel_compose(X, Y, kernel_spec)
    payload = result.payload
    assert "estimate" in payload
    assert "K" in payload
    assert "diagonal" in payload
    assert "min_eigenvalue" in payload
    assert "is_psd" in payload
    assert "n" in payload
    assert "method" in payload
    K = payload["K"]
    assert len(K) == n
    assert all(len(row) == n for row in K)
    assert len(payload["diagonal"]) == n
    assert payload["n"] == n
    assert payload["is_psd"] in (0, 1)


def test_gpkern_edge():
    """Test edge case: default kernel spec with Y=None."""
    rng = np.random.default_rng(42)
    n, d = 40, 3
    X = rng.normal(0, 1, (n, d))
    # Default kernel_spec, Y=None: min_eigenvalue is computed
    result = gp_kernel_compose(X)
    payload = result.payload
    assert "K" in payload
    assert "estimate" in payload
    assert "min_eigenvalue" in payload
    K = payload["K"]
    assert len(K) == n
    assert all(len(row) == n for row in K)
    assert payload["n"] == n
    # With Y=None, min_eigenvalue is computed via jacobi, so it's finite
    assert math.isfinite(payload["min_eigenvalue"])
