"""Tests for hrzplri.horowitz_plr_identification."""

from morie.fn import _array_core as np

from morie.fn.hrzplri import horowitz_plr_identification


def test_hrzplri_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_plr_identification(x, z)
    assert isinstance(result, dict)
    assert "identified" in result


def test_hrzplri_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_plr_identification(x, z)
    assert isinstance(result, dict)
