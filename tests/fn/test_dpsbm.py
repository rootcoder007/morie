"""Tests for dpsbm.dp_stochastic_block."""

from morie.fn import _array_core as np

from morie.fn.dpsbm import dp_stochastic_block


def test_dpsbm_basic():
    """Test basic functionality."""
    adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    alpha = 0.05
    result = dp_stochastic_block(adjacency, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == 3
    assert result["n_blocks"] >= 1
    assert result["n_blocks"] == result["estimate"]
    assert len(result["z"]) == 3
    assert len(result["counts"]) == result["n_blocks"]


def test_dpsbm_edge():
    """Test edge cases."""
    adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    alpha = 0.05
    result = dp_stochastic_block(adjacency, alpha)
    assert isinstance(result, dict)
    assert "z" in result
    assert "counts" in result
    assert "n_blocks" in result
    assert "log_likelihood" in result
    assert "n" in result
    assert sum(result["counts"]) == 3
