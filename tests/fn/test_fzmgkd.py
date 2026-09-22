"""Tests for fzmgkd.fauzi_modified_gamma_kde."""

from morie.fn import _array_core as np

from morie.fn.fzmgkd import fauzi_modified_gamma_kde


def test_fzmgkd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.5, 5.0, 100)
    bandwidth = 0.3
    grid = rng.uniform(0.5, 5.0, 100)
    result = fauzi_modified_gamma_kde(x, grid, bandwidth)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "ah" in result
    assert "a4h" in result
    assert "grid" in result
    assert "h" in result
    assert "n" in result
    assert "method" in result
    # The estimate is non-negative by construction (Eq. 1.14).
    for v in result["estimate"]:
        assert v >= 0.0


def test_fzmgkd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.5, 5.0, 100)
    bandwidth = 0.3
    grid = rng.uniform(0.5, 5.0, 100)
    result = fauzi_modified_gamma_kde(x, grid, bandwidth)
    assert isinstance(result, dict)
    assert len(result["estimate"]) == len(result["grid"]) == grid.size
    assert result["n"] == x.size
    assert result["h"] == float(bandwidth)
