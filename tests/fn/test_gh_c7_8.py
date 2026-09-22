"""Tests for gh_c7_8.ghosal_loc_semipara."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_8 import ghosal_loc_semipara


def test_gh_c7_8_basic():
    """Test basic functionality with documented defaults."""
    theta0 = 0.7
    result = ghosal_loc_semipara(theta0=theta0, n=400, seed=42)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # Independent expected estimate: profile the same seed-replicated data
    # over the documented grid using the documented symmetrized histogram
    # likelihood.
    rng = np.random.default_rng(42)
    data = [theta0 + float(rng.normal(0, 1)) for _ in range(400)]

    def loglik(th):
        k = 16
        counts = [1.0] * k
        for r in data:
            rr = min(max(r - th, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            counts[idx] += 0.5
            counts[k - 1 - idx] += 0.5
        tot = sum(counts)
        ll = 0.0
        for r in data:
            rr = min(max(r - th, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            ll += float(np.log(float(counts[idx] / tot / 0.5)))
        return ll

    grid = [theta0 - 1.0 + 2.0 * j / 40 for j in range(41)]
    lls = [loglik(t) for t in grid]
    expected_estimate = grid[lls.index(max(lls))]
    assert np.all(np.isfinite(np.asarray([estimate], dtype=float)))
    assert abs(estimate - expected_estimate) < 1e-12


def test_gh_c7_8_edge():
    """Test edge case: smallest sample size n=1."""
    theta0 = 42.0
    result = ghosal_loc_semipara(theta0=theta0, n=1, seed=42)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # Independent expected estimate under the same formula with n=1.
    rng = np.random.default_rng(42)
    data = [theta0 + float(rng.normal(0, 1)) for _ in range(1)]

    def loglik(th):
        k = 16
        counts = [1.0] * k
        for r in data:
            rr = min(max(r - th, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            counts[idx] += 0.5
            counts[k - 1 - idx] += 0.5
        tot = sum(counts)
        ll = 0.0
        for r in data:
            rr = min(max(r - th, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            ll += float(np.log(float(counts[idx] / tot / 0.5)))
        return ll

    grid = [theta0 - 1.0 + 2.0 * j / 40 for j in range(41)]
    lls = [loglik(t) for t in grid]
    expected_estimate = grid[lls.index(max(lls))]
    assert abs(estimate - expected_estimate) < 1e-12
