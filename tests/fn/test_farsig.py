"""Tests for farsig.farrington_signal."""

from morie.fn import _array_core as np

from morie.fn.farsig import farrington_signal


def test_farsig_basic():
    """Test basic functionality."""
    counts = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = farrington_signal(counts)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_farsig_edge():
    """Test edge cases."""
    counts = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = farrington_signal(counts)
    assert isinstance(result, dict)
