"""Tests for hmnadm.geron_nadam."""

from morie.fn import _array_core as np

from morie.fn.hmnadm import geron_nadam


def test_hmnadm_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_nadam(grads)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmnadm_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_nadam(grads)
    assert isinstance(result, dict)
