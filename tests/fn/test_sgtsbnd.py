"""Tests for sgtsbnd.sgt_sbm_detect_threshold."""

from morie.fn import _array_core as np

from morie.fn.sgtsbnd import sgt_sbm_detect_threshold


def test_sgtsbnd_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    result = sgt_sbm_detect_threshold(a, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "detectable" in result


def test_sgtsbnd_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    result = sgt_sbm_detect_threshold(a, b)
    assert isinstance(result, dict)
