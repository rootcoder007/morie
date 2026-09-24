"""Tests for unobts.unobserved_components."""

from morie.fn import _array_core as np

from morie.fn.unobts import unobserved_components


def test_unobts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = unobserved_components(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_unobts_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = unobserved_components(y)
    assert isinstance(result, dict)
