"""Tests for alpz.alphazero_search."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.alpz import alphazero_search


def test_alpz_basic():
    """Test basic functionality."""
    def net(s):
        # Return 3 actions with priors and a value
        return [0.5, 0.3, 0.2], 0.0

    state = 0
    num_sim = 20
    result = alphazero_search(state, net, num_sim)
    assert isinstance(result, dict)
    # Check all documented keys are present
    assert "estimate" in result
    assert "pi" in result
    assert "n" in result
    assert "q" in result
    assert "p" in result
    assert "value" in result
    assert "n_nodes" in result
    # estimate is a valid action index
    assert 0 <= result["estimate"] < 3
    # value is finite
    assert math.isfinite(result["value"])
    # root was expanded at least once
    assert result["n_nodes"] >= 1


def test_alpz_edge():
    """Test edge cases with terminal function and custom parameters."""
    def net(s):
        return [0.6, 0.4], 0.5

    def terminal(s):
        return False

    state = 42
    num_sim = 10
    result = alphazero_search(state, net, num_sim, terminal=terminal, c_puct=2.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "pi" in result
    assert "value" in result
    assert "n_nodes" in result
    assert 0 <= result["estimate"] < 2
    assert math.isfinite(result["value"])
    assert result["n_nodes"] >= 1
