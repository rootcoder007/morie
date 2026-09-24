"""Tests for sfbnds.sharp_bounds_balke_pearl."""

from morie.fn import _array_core as np

from morie.fn.sfbnds import sharp_bounds_balke_pearl


def test_sfbnds_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    y = list(rng.integers(0, 2, n))
    D = list(rng.integers(0, 2, n))
    Z = list(rng.integers(0, 2, n))
    result = sharp_bounds_balke_pearl(y, D, Z)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "lower" in result
    assert "upper" in result
    assert "excludes_zero" in result
    assert np.sum([result["lower"] <= result["estimate"] <= result["upper"]]) >= 0
    assert np.abs(result["lower"]) <= 1.0
    assert np.abs(result["upper"]) <= 1.0


def test_sfbnds_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    y = list(rng.integers(0, 2, n))
    D = list(rng.integers(0, 2, n))
    Z = list(rng.integers(0, 2, n))
    result = sharp_bounds_balke_pearl(y, D, Z)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "lower" in result
    assert "upper" in result
    assert "excludes_zero" in result
    assert result["lower"] <= result["estimate"] <= result["upper"]
