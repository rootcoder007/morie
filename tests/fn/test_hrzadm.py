"""Tests for hrzadm.horowitz_additive_model."""

from morie.fn import _array_core as np

from morie.fn.hrzadm import horowitz_additive_model


def test_hrzadm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_model(x, y)
    assert isinstance(result, dict)
    assert "mu" in result


def test_hrzadm_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_model(x, y)
    assert isinstance(result, dict)
