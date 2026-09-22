"""Tests for evgevs.evt_gev_sample."""

from morie.fn import _array_core as np

from morie.fn.evgevs import evt_gev_sample


def test_evgevs_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = 0.1
    n = 100
    result = evt_gev_sample(n, mu, sigma, xi, seed=42)
    assert isinstance(result, dict)
    # Independent check via inverse-CDF formula (Coles 2001 eq. 3.4):
    # G^{-1}(u) = mu + (sigma/xi) * ((-log(u))^{-xi} - 1)
    rng = np.random.default_rng(42)
    u = rng.random(n)
    if xi == 0.0:
        expected = mu - sigma * np.log(u)
    else:
        expected = mu + (sigma / xi) * (np.power(-np.log(u), -xi) - 1.0)
    assert np.allclose(result["x"], expected)
    assert result["n"] == n
    assert result["method"] == "GEV inverse-CDF sampler (Coles 2001 eq. 3.4)"


def test_evgevs_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = 0.1
    n = 100
    result = evt_gev_sample(n, mu, sigma, xi, seed=42)
    assert isinstance(result, dict)
