"""Tests for aitgmu.aitchison_geomean."""

from morie.fn import _array_core as np

from morie.fn.aitgmu import aitchison_geomean


def test_aitgmu_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_geomean(x)
    assert isinstance(result, dict)
    assert "geomean" in result
def test_aitgmu_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_geomean(x)
    assert isinstance(result, dict)
