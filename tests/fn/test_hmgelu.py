"""Tests for hmgelu.geron_gelu."""

from morie.fn import _array_core as np

from morie.fn.hmgelu import geron_gelu


def test_hmgelu_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_gelu(z)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmgelu_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_gelu(z)
    assert isinstance(result, dict)
