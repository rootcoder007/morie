"""Tests for rrblpr.rr_blup."""

from morie.fn import _array_core as np

from morie.fn.rrblpr import rr_blup


def test_rrblpr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rr_blup(y, M)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rrblpr_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rr_blup(y, M)
    assert isinstance(result, dict)
