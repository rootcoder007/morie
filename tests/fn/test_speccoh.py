"""Tests for speccoh.coherence."""

from morie.fn import _array_core as np

from morie.fn.speccoh import coherence


def test_speccoh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = coherence(x, y)
    assert isinstance(result, dict)
    assert "omega" in result


def test_speccoh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = coherence(x, y)
    assert isinstance(result, dict)
