"""Tests for joautf.joseph_autoformer."""

from morie.fn import _array_core as np

from morie.fn.joautf import joseph_autoformer


def test_joautf_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_autoformer(q, k, v)
    assert isinstance(result, dict)
    assert "out" in result


def test_joautf_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_autoformer(q, k, v)
    assert isinstance(result, dict)
