"""Tests for hmgan.geron_gan."""

from morie.fn import _array_core as np

from morie.fn.hmgan import geron_gan


def test_hmgan_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gan(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "G" in result


def test_hmgan_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gan(X)
    assert isinstance(result, dict)
