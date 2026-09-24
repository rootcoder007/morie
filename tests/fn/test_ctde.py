"""Tests for ctde.controlled_direct_effect."""

import math

from morie.fn import _array_core as np

from morie.fn.ctde import controlled_direct_effect


def test_ctde_basic():
    """Test basic functionality."""
    n = 40
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, n)
    M = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    m = 0.0
    result = controlled_direct_effect(X, M, Y, m)
    payload = getattr(result, "payload", None)
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert math.isfinite(payload["estimate"])
    assert payload["m"] == m
    assert payload["a"] == 1.0
    assert payload["astar"] == 0.0
    assert payload["n"] == n


def test_ctde_edge():
    """Test edge cases."""
    n = 40
    p = 3
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, n)
    M = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    C = rng.normal(0, 1, (n, p))
    m = 0.5
    result = controlled_direct_effect(X, M, Y, m, C=C, a=1.0, astar=0.0)
    payload = getattr(result, "payload", None)
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert math.isfinite(payload["estimate"])
    assert payload["a"] == 1.0
    assert payload["astar"] == 0.0
    assert payload["m"] == m
    assert payload["n"] == n
