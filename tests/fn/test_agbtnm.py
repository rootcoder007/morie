"""Tests for agbtnm.alphazero_batch_norm."""

from morie.fn import _array_core as np

from morie.fn.agbtnm import alphazero_batch_norm


def test_agbtnm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x)
    assert isinstance(result, dict)
    assert "runmean" in result
def test_agbtnm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x)
    assert isinstance(result, dict)
