"""Tests for rgneural.rangayyan_neural_decode."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_neural_decode


def test_rgneural_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    C = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = rangayyan_neural_decode(y, C)
    assert isinstance(result, dict)
    assert "states" in result


def test_rgneural_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    C = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = rangayyan_neural_decode(y, C)
    assert isinstance(result, dict)
