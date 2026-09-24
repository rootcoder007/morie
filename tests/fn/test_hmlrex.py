"""Tests for hmlrex.geron_lr_exponential."""

from morie.fn import _array_core as np

from morie.fn.hmlrex import geron_lr_exponential


def test_hmlrex_basic():
    """Test basic functionality."""
    eta0 = 0.5
    decay = 0.5
    t = 0.5
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmlrex_edge():
    """Test edge cases."""
    eta0 = 0.5
    decay = 0.5
    t = 0.5
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)
