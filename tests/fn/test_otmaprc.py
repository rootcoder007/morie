"""Tests for otmaprc.ot_map_recovery_brenier."""

from morie.fn import _array_core as np

from morie.fn.otmaprc import ot_map_recovery_brenier


def test_otmaprc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ot_map_recovery_brenier(x, y)
    assert isinstance(result, dict)
    assert "map" in result


def test_otmaprc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ot_map_recovery_brenier(x, y)
    assert isinstance(result, dict)
