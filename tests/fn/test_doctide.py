"""Tests for doctide.de_chaisemartin_dhaultfoeuille."""

from morie.fn import _array_core as np

from morie.fn.doctide import de_chaisemartin_dhaultfoeuille


def test_doctide_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_unit = np.random.default_rng(41)
    n = 100
    y = rng_y.normal(0, 1, n)
    D = (rng_D.uniform(0, 1, n) > 0.5).astype("float64")
    unit = rng_unit.integers(0, 10, n).astype("float64")
    time = np.linspace(0, 9, n)
    result = de_chaisemartin_dhaultfoeuille(y, D, unit, time)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "periods" in result
    assert "did_plus" in result
    assert "did_minus" in result
    assert "n10" in result
    assert "n01" in result
    assert "n_switch" in result
    assert "n_units" in result
    assert "n" in result


def test_doctide_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_unit = np.random.default_rng(41)
    n = 100
    y = rng_y.normal(0, 1, n)
    D = (rng_D.uniform(0, 1, n) > 0.5).astype("float64")
    unit = rng_unit.integers(0, 10, n).astype("float64")
    time = np.linspace(0, 9, n)
    result = de_chaisemartin_dhaultfoeuille(y, D, unit, time)
    assert isinstance(result, dict)
    assert result["n"] == 100
