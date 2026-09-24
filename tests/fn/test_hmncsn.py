"""Tests for hmncsn.geron_ncsn."""

from morie.fn import _array_core as np

from morie.fn.hmncsn import geron_ncsn


def test_hmncsn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_ncsn(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "models" in result


def test_hmncsn_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_ncsn(X)
    assert isinstance(result, dict)
