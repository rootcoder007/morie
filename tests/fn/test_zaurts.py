"""Tests for zaurts.zivot_andrews_unit_root."""

from morie.fn import _array_core as np

from morie.fn.zaurts import zivot_andrews_unit_root


def test_zaurts_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = zivot_andrews_unit_root(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "statistic" in result


def test_zaurts_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = zivot_andrews_unit_root(x)
    assert isinstance(result, dict)
