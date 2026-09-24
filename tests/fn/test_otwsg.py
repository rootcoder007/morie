"""Tests for otwsg.ot_wasserstein_gauss."""

import math

from morie.fn import _array_core as np

from morie.fn.otwsg import ot_wasserstein_gauss


def _make_psd(rng, d):
    """Build a d x d symmetric positive (semi-)definite matrix L L^T."""
    L = rng.normal(0, 1, (d, d))
    return [[sum(L[i][k] * L[j][k] for k in range(d)) for j in range(d)]
            for i in range(d)]


def test_otwsg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d = 4
    mu1 = rng.normal(0, 1, d)
    mu2 = rng.normal(0, 1, d)
    Sigma1 = _make_psd(rng, d)
    Sigma2 = _make_psd(rng, d)
    result = ot_wasserstein_gauss(mu1, Sigma1, mu2, Sigma2)
    assert isinstance(result, dict)
    for key in ("W2", "W2_sq", "mean_part", "bures_sq", "d"):
        assert key in result
    assert math.isfinite(result["W2"])
    assert math.isfinite(result["W2_sq"])
    assert math.isfinite(result["mean_part"])
    assert math.isfinite(result["bures_sq"])
    assert result["W2"] >= 0
    assert result["W2_sq"] >= 0
    assert result["d"] == d


def test_otwsg_edge():
    """Test edge case - identical distributions give zero distance."""
    rng = np.random.default_rng(42)
    d = 3
    mu = rng.normal(0, 1, d)
    Sigma = _make_psd(rng, d)
    result = ot_wasserstein_gauss(mu, Sigma, mu, Sigma)
    assert isinstance(result, dict)
    assert "W2" in result
    assert "W2_sq" in result
    assert "mean_part" in result
    assert "bures_sq" in result
    assert "d" in result
    assert math.isfinite(result["W2"])
    assert result["W2"] >= 0
    assert result["mean_part"] < 1e-9
    assert result["W2"] < 1e-6
    assert result["d"] == d
