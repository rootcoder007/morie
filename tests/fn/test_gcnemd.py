"""Tests for gcnemd.gcn."""

import math

from morie.fn import _array_core as np

from morie.fn.gcnemd import gcn


def test_gcnemd_basic():
    """Test basic functionality."""
    n = 10
    p = 5
    q = 3
    rng = np.random.default_rng(42)
    G = np.eye(n)
    X = rng.normal(0, 1, (n, p))
    W = rng.normal(0, 1, (p, q))
    result = gcn(G, X, W)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "H" in result
    assert "n" in result
    assert result["n"] == n
    assert math.isfinite(result["estimate"])


def test_gcnemd_edge():
    """Test edge cases."""
    n = 5
    p = 4
    q = 2
    rng = np.random.default_rng(42)
    G = np.eye(n)
    X = rng.normal(0, 1, (n, p))
    W = rng.normal(0, 1, (p, q))
    result = gcn(G, X, W)
    assert isinstance(result, dict)
    assert "H" in result
    assert len(result["H"]) == n
    assert len(result["H"][0]) == q
    assert math.isfinite(result["estimate"])
