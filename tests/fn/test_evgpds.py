"""Tests for evgpds.evt_gpd_sample."""

from morie.fn import _array_core as np

from morie.fn.evgpds import evt_gpd_sample


def test_evgpds_basic():
    """Test basic functionality."""
    sigma = 1.0
    xi = 0.1
    n = 100
    result = evt_gpd_sample(n, sigma, xi)
    assert isinstance(result, dict)
    assert "y" in result
    assert "n" in result
    assert result["n"] == n
    assert len(result["y"]) == n
    # All sampled excesses should be >= 0 for a GPD (threshold excesses).
    assert np.all(np.asarray(result["y"]) >= 0)


def test_evgpds_edge():
    """Test edge cases."""
    sigma = 2.5
    xi = 0.0
    n = 50
    result = evt_gpd_sample(n, sigma, xi, seed=7)
    assert isinstance(result, dict)
    assert "y" in result
    assert len(result["y"]) == n
    # For xi=0 (exponential tail), the CDF inverse is
    # F^{-1}(u) = -sigma * log(1 - u).
    # Equivalently, exp(-y/sigma) ~ U(0,1), so y/sigma ~ Exp(1).
    rng = np.random.default_rng(7)
    u = rng.uniform(0.0, 1.0, n)
    expected = -sigma * np.log(1.0 - u)
    assert np.allclose(np.sort(result["y"]), np.sort(expected))
