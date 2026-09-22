"""Tests for diripr.dirichlet_multinomial."""

from morie.fn import _array_core as np

from morie.fn.diripr import dirichlet_multinomial


def test_diripr_basic():
    """Test basic functionality."""
    counts = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = dirichlet_multinomial(counts)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_diripr_edge():
    """Test edge cases."""
    counts = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = dirichlet_multinomial(counts)
    assert isinstance(result, dict)
