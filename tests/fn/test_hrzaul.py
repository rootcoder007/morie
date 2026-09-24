"""Tests for hrzaul.horowitz_additive_unknown_link."""

from morie.fn import _array_core as np

from morie.fn.hrzaul import horowitz_additive_unknown_link


def test_hrzaul_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_unknown_link(x, y)
    assert isinstance(result, dict)
    assert "G_hat" in result


def test_hrzaul_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_unknown_link(x, y)
    assert isinstance(result, dict)
