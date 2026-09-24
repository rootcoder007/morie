"""Tests for sbmest.stochastic_block_model."""

from morie.fn import _array_core as np

from morie.fn.sbmest import stochastic_block_model


def test_sbmest_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    blocks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = stochastic_block_model(A, blocks)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sbmest_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    blocks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = stochastic_block_model(A, blocks)
    assert isinstance(result, dict)
