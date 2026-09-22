"""Tests for dpmem.dirichlet_process_mixture."""

from morie.fn import _array_core as np

from morie.fn.dpmem import dirichlet_process_mixture


def test_dpmem_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dirichlet_process_mixture(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpmem_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dirichlet_process_mixture(y)
    assert isinstance(result, dict)
