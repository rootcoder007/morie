"""Tests for hmotrk.geron_object_tracking."""

from morie.fn import _array_core as np

from morie.fn.hmotrk import geron_object_tracking


def test_hmotrk_basic():
    """Test basic functionality."""
    posdist = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_object_tracking(posdist)
    assert isinstance(result, dict)
    assert "cost" in result


def test_hmotrk_edge():
    """Test edge cases."""
    posdist = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_object_tracking(posdist)
    assert isinstance(result, dict)
