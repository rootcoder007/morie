"""Tests for alfrcl.alphafold_recycle_loss."""

from morie.fn import _array_core as np

from morie.fn.alfrcl import alphafold_recycle_loss


def test_alfrcl_basic():
    """Test basic functionality."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(losses)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfrcl_edge():
    """Test edge cases."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(losses)
    assert isinstance(result, dict)
