"""Tests for moeswt.moe_switch_routing."""

import math

from morie.fn import _array_core as np
from morie.fn.moeswt import moe_switch_routing


def test_moeswt_basic():
    """Test basic functionality with x overriding y."""
    T = 40  # number of tokens
    p = 3   # features per token
    N = 4   # number of experts

    y = np.random.default_rng(43).normal(0, 1, (T, p))
    x = np.random.default_rng(42).normal(0, 1, (T, p))
    W_g = np.random.default_rng(7).normal(0, 1, (p, N))

    result = moe_switch_routing(y, x, W_g, capacity=1.25)

    assert isinstance(result, dict)
    # Verify all keys named in the return statement exist.
    assert "estimate" in result
    assert "aux_loss" in result
    assert "assign" in result
    assert "dropped" in result
    assert "out" in result
    # The two auxiliary-loss aliases must agree and be finite.
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["aux_loss"])
    assert result["estimate"] == result["aux_loss"]
    # One expert index per token, either -1 (dropped) or in [0, N).
    assert len(result["assign"]) == T
    for a in result["assign"]:
        assert -1 <= a < N
    # dropped is a non-negative integer.
    assert isinstance(result["dropped"], int)
    assert result["dropped"] >= 0
    # out has T rows.
    assert len(result["out"]) == T


def test_moeswt_edge():
    """Test edge case: x=None path uses y, with explicit alpha."""
    T = 8
    p = 3
    N = 2

    y = np.random.default_rng(43).normal(0, 1, (T, p))
    W_g = np.random.default_rng(7).normal(0, 1, (p, N))

    # Only pass y (no x), exercising the fallback path.
    result = moe_switch_routing(y=y, W_g=W_g, capacity=2.0, alpha=0.05)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "aux_loss" in result
    assert "assign" in result
    assert "dropped" in result
    assert "out" in result
    assert math.isfinite(result["estimate"])
    assert len(result["assign"]) == T
    assert result["dropped"] >= 0
    assert len(result["out"]) == T
