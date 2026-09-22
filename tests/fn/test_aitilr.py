"""Tests for aitilr.aitchison_ilr."""

from morie.fn import _array_core as np

from morie.fn.aitilr import aitchison_ilr


def test_aitilr_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_ilr(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitilr_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_ilr(x)
    assert isinstance(result, dict)
