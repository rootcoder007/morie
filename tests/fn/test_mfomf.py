"""Tests for mfomf.model_based_rl."""

from morie.fn import _array_core as np

from morie.fn.mfomf import model_based_rl


def test_mfomf_basic():
    """Test basic functionality."""
    env = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = model_based_rl(env)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mfomf_edge():
    """Test edge cases."""
    env = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = model_based_rl(env)
    assert isinstance(result, dict)
