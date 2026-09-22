"""Tests for agdrcn.alphazero_dirichlet_concentration."""

from morie.fn import _array_core as np

from morie.fn.agdrcn import alphazero_dirichlet_concentration


def test_agdrcn_basic():
    """Test basic functionality."""
    avg_legal = 0.0
    scale = 0.0
    result = alphazero_dirichlet_concentration(avg_legal, scale)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_agdrcn_edge():
    """Test edge cases."""
    avg_legal = 0.0
    scale = 0.0
    result = alphazero_dirichlet_concentration(avg_legal, scale)
    assert isinstance(result, dict)
