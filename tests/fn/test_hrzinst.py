"""Tests for hrzinst.horowitz_instruments_transformation."""

from morie.fn import _array_core as np

from morie.fn.hrzinst import horowitz_instruments_transformation


def test_hrzinst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_instruments_transformation(X, Z)
    assert isinstance(result, dict)
    assert "first_stage_r2" in result


def test_hrzinst_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_instruments_transformation(X, Z)
    assert isinstance(result, dict)
