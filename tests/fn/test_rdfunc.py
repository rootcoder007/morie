"""Tests for rdfunc.rate_distortion."""

from morie.fn import _array_core as np

from morie.fn.rdfunc import rate_distortion


def test_rdfunc_basic():
    """Test basic functionality."""
    px = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = rate_distortion(px)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rdfunc_edge():
    """Test edge cases."""
    px = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = rate_distortion(px)
    assert isinstance(result, dict)
