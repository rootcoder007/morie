"""Tests for gatemd.graph_attention_net."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.gatemd import graph_attention_net


def test_gatemd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    p = 3
    G = np.eye(n)
    X = rng.normal(0, 1, (n, p))
    result = graph_attention_net(G, X, heads=1)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])


def test_gatemd_edge():
    """Test that heads must be at least 1."""
    rng = np.random.default_rng(42)
    n = 10
    p = 3
    G = np.eye(n)
    X = rng.normal(0, 1, (n, p))
    with pytest.raises(ValueError):
        graph_attention_net(G, X, heads=0)
