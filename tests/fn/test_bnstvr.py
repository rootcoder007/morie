"""Tests for bnstvr.bound_treatment_variation."""

from morie.fn import _array_core as np

from morie.fn.bnstvr import bound_treatment_variation


def test_bnstvr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    y = rng.normal(0, 1, n)
    rng_d = np.random.default_rng(42)
    D = rng_d.integers(0, 2, n)
    rng_x = np.random.default_rng(41)
    X = rng_x.integers(0, 5, n)
    result = bound_treatment_variation(y, D, X)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert "estimate" in result
    assert "n_cells" in result
    assert "refuted" in result
    assert "n" in result
    assert result["n"] == n


def test_bnstvr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 40
    y = rng.normal(0, 1, n)
    rng_d = np.random.default_rng(42)
    D = rng_d.integers(0, 2, n)
    rng_x = np.random.default_rng(41)
    X = rng_x.integers(0, 3, n)
    result = bound_treatment_variation(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
