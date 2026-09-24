"""Tests for poltre.polya_tree_prior."""

from morie.fn import _array_core as np

from morie.fn.poltre import polya_tree_prior


def test_poltre_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polya_tree_prior(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_poltre_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polya_tree_prior(x)
    assert isinstance(result, dict)
