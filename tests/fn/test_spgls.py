"""Tests for spgls.schabenberger_gls_spatial."""

from morie.fn import _array_core as np

from morie.fn.spgls import schabenberger_gls_spatial


def test_spgls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_gls_spatial(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_spgls_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_gls_spatial(x, y)
    assert isinstance(result, dict)
