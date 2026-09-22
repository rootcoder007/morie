"""Tests for agdirn.alphazero_dirichlet_noise."""

from morie.fn import _array_core as np

from morie.fn.agdirn import alphazero_dirichlet_noise


def test_agdirn_basic():
    """Test basic functionality."""
    p = 5
    result = alphazero_dirichlet_noise(p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_agdirn_edge():
    """Test edge cases."""
    p = 5
    result = alphazero_dirichlet_noise(p)
    assert isinstance(result, dict)
