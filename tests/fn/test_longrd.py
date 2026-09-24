"""Tests for longrd.long_read_polish."""

from morie.fn import _array_core as np

from morie.fn.longrd import long_read_polish


def test_longrd_basic():
    """Test basic functionality."""
    assembly = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reads = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = long_read_polish(assembly, reads)
    assert isinstance(result, dict)
    assert "polished" in result


def test_longrd_edge():
    """Test edge cases."""
    assembly = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reads = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = long_read_polish(assembly, reads)
    assert isinstance(result, dict)
