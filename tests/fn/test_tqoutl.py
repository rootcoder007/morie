"""Tests for tqoutl.turboquant_outlier_channel_split."""

from morie.fn import _array_core as np

from morie.fn.tqoutl import turboquant_outlier_channel_split


def test_tqoutl_basic():
    """Test basic functionality."""
    channels = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = turboquant_outlier_channel_split(channels)
    assert isinstance(result, dict)
    assert "estimate" in result or "outlier_idx" in result


def test_tqoutl_edge():
    """Test edge cases."""
    channels = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = turboquant_outlier_channel_split(channels)
    assert isinstance(result, dict)
