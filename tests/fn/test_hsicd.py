"""Tests for hsicd.hsic_independence."""

from morie.fn import _array_core as np

from morie.fn.hsicd import hsic_independence


def test_hsicd_basic():
    """Test basic functionality."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    b = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = hsic_independence(a, b)
    assert isinstance(result, dict)
    assert "hsic" in result or "hsic" in result


def test_hsicd_edge():
    """Test edge cases."""
    a = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    b = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = hsic_independence(a, b)
    assert isinstance(result, dict)
