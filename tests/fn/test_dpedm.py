"""Tests for dpedm.approx_dp."""

from morie.fn import _array_core as np

from morie.fn.dpedm import approx_dp


def test_dpedm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    D = np.zeros(10)
    D_prime = np.r_[np.zeros(9), 1.0]
    sigma = 3.0
    mech = lambda X, rng: float(np.asarray(X, dtype=float).sum()
                                + rng.normal(0.0, sigma))
    result = approx_dp(mech, D, D_prime, epsilon=1.0,
                       n_samples=20000, bins=50, seed=0)
    assert isinstance(result, dict)
    assert "delta_empirical" in result
    assert "epsilon" in result
    assert "n_violating_bins" in result
    assert result["epsilon"] == 1.0
    # Generously noised Gaussian at eps=1 should need very little delta.
    assert 0.0 <= result["delta_empirical"] < 0.05


def test_dpedm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    D = np.zeros(10)
    D_prime = np.r_[np.zeros(9), 1.0]
    weak = lambda X, rng: float(np.asarray(X, dtype=float).sum()
                                + rng.normal(0.0, 0.05))
    # Tiny noise: at eps=1 this mechanism cannot satisfy the bound on a
    # substantial share of its output range.
    r1 = approx_dp(weak, D, D_prime, epsilon=1.0,
                   n_samples=20000, bins=50, seed=0)
    assert isinstance(r1, dict)
    assert r1["delta_empirical"] > 0.2
    # Raising epsilon lowers the required delta.
    r_lo = approx_dp(weak, D, D_prime, epsilon=0.5,
                     n_samples=20000, bins=50, seed=0)
    r_hi = approx_dp(weak, D, D_prime, epsilon=8.0,
                     n_samples=20000, bins=50, seed=0)
    assert r_hi["delta_empirical"] <= r_lo["delta_empirical"]
    # Deterministic mechanism -> delta_empirical collapses to 1.0 with a warning.
    det = lambda X, rng: float(np.asarray(X, dtype=float).sum())
    r_det = approx_dp(det, D, D_prime, epsilon=1.0,
                      n_samples=2000, bins=10, seed=0)
    assert r_det["delta_empirical"] == 1.0
