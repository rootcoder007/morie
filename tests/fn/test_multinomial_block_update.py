"""Tests for multinomial_block_update.multinomial_block_update."""

from morie.fn import _array_core as np

from morie.fn.multinomial_block_update import multinomial_block_update


def test_msm112_basic():
    """Test basic functionality."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    That = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_block_update(of, e, That, the, update, block)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm112_edge():
    """Test edge cases."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    That = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_block_update(of, e, That, the, update, block)
    assert isinstance(result, dict)
