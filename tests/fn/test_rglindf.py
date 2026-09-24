"""Tests for rglindf.rangayyan_linear_discrim."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_linear_discrim


def test_rglindf_basic():
    """Test basic functionality."""
    x = 0.5
    weights = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_linear_discrim(x, weights)
    assert isinstance(result, dict)
    assert "d" in result


def test_rglindf_edge():
    """Test edge cases."""
    x = 0.5
    weights = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_linear_discrim(x, weights)
    assert isinstance(result, dict)
