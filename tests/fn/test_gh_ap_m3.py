"""Tests for gh_ap_m3.ghosal_slice_sampler."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_m3 import ghosal_slice_sampler


def test_gh_ap_m3_basic():
    """Test basic functionality."""
    result = ghosal_slice_sampler()
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_ap_m3_seeded_reproducible():
    """Same seed reproduces the same estimate."""
    r1 = ghosal_slice_sampler(n_draws=2000, seed=42)
    r2 = ghosal_slice_sampler(n_draws=2000, seed=42)
    assert r1["estimate"] == r2["estimate"]


def test_gh_ap_m3_mean_close_to_target():
    """Sample mean of Exp(1) draws should be near 1.0."""
    n = 50000
    res = ghosal_slice_sampler(n_draws=n, seed=123)
    # Independent Monte-Carlo expectation: 95% CLT half-width for Exp(1) is ~1.96/sqrt(n)
    half_width = 1.96 / (n ** 0.5)
    assert abs(res["estimate"] - 1.0) < half_width
    assert abs(res["target_mean"] - 1.0) < 1e-12
    assert abs(res["gap"] - abs(res["estimate"] - 1.0)) < 1e-12


def test_gh_ap_m3_keys_and_method():
    """Result payload has the documented keys."""
    res = ghosal_slice_sampler(n_draws=100, seed=7)
    for key in ("estimate", "target_mean", "gap", "method"):
        assert key in res
    assert isinstance(res["method"], str)
    assert "slice" in res["method"].lower()
