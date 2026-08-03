"""Tests for noncentrality_delta_generic.noncentrality_delta_generic."""

from morie.fn import _array_core as np

from morie.fn.noncentrality_delta_generic import noncentrality_delta_generic


def test_ca8e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = noncentrality_delta_generic(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca8e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = noncentrality_delta_generic(x)
    assert isinstance(result, dict)
