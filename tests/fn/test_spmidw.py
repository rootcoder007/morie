"""Tests for spmidw.schabenberger_idw."""

from morie.fn import _array_core as np

from morie.fn.spmidw import schabenberger_idw


def test_spmidw_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_idw(coords, z, target)
    assert isinstance(result, dict)
    assert "prediction" in result


def test_spmidw_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_idw(coords, z, target)
    assert isinstance(result, dict)
