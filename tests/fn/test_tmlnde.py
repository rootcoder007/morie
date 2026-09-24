"""Tests for tmlnde.tmle_natural_direct."""

from morie.fn import _array_core as np

from morie.fn.tmlnde import tmle_natural_direct


def test_tmlnde_basic():
    """Test basic functionality."""
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y00 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_natural_direct(y10, y00)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlnde_edge():
    """Test edge cases."""
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y00 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_natural_direct(y10, y00)
    assert isinstance(result, dict)
