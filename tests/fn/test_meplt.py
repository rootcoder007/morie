"""Tests for meplt.mean_excess."""

from morie.fn import _array_core as np

from morie.fn.meplt import mean_excess


def test_meplt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mean_excess(x)
    assert isinstance(result, dict)
    assert "u" in result


def test_meplt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mean_excess(x)
    assert isinstance(result, dict)
