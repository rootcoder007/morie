"""Tests for gh_c13_12.ghosal_smhaz_gp."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_12 import ghosal_smhaz_gp


def test_gh_c13_12_basic():
    """Test basic functionality."""
    result = ghosal_smhaz_gp()
    assert "estimate" in result
    est = result["estimate"]
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))


def test_gh_c13_12_n():
    """Test that n is passed through correctly."""
    result = ghosal_smhaz_gp(n=200)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c13_12_seed_reproducible():
    """Test that the same seed gives the same estimate."""
    r1 = ghosal_smhaz_gp(n=500, seed=42)
    r2 = ghosal_smhaz_gp(n=500, seed=42)
    assert r1["estimate"] == r2["estimate"]


def test_gh_c13_12_constant_hazard_recovery():
    """The true hazard is constant (1.0); the binned estimator should recover it."""
    import math

    k = 6
    n = 5000
    seed = 42
    rng = np.random.default_rng(seed)
    lam0 = 1.0

    d_ = [0.0] * k
    e_ = [0.0] * k
    for _ in range(n):
        x = -math.log(max(float(rng.uniform(0, 1)), 1e-12)) / lam0
        for b in range(k):
            lo, hi = b * 0.3, (b + 1) * 0.3
            if x >= hi:
                e_[b] += 0.3
            elif x > lo:
                e_[b] += x - lo
                d_[b] += 1.0
                break
    f = [math.log(max((d + 0.5) / (e + 0.5), 1e-6))
         for d, e in zip(d_, e_)]
    haz = [math.exp(v) for v in f]
    expected_err = sum(abs(h - lam0) for h in haz) / k

    result = ghosal_smhaz_gp(n=n, seed=seed)
    assert abs(result["estimate"] - expected_err) < 1e-12
    assert result["method"] == "GP smooth hazard (GvdV 2017 sec. 13.5)"
    assert "hazard_by_bin" in result
