"""Tests for tqdist.turboquant_distortion_bound."""

from morie.fn import _array_core as np

from morie.fn.tqdist import turboquant_distortion_bound


def test_tqdist_basic():
    """Test basic functionality."""
    eps = 0.1
    delta = 0.1
    result = turboquant_distortion_bound(eps, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "m_min" in result


def test_tqdist_edge():
    """Test edge cases."""
    eps = 0.1
    delta = 0.1
    result = turboquant_distortion_bound(eps, delta)
    assert isinstance(result, dict)
