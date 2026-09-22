"""Tests for gh_c12_5.ghosal_eff_infl_fn."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_5 import ghosal_eff_infl_fn


def test_gh_c12_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = 3.0
    result = ghosal_eff_infl_fn(x, t)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(estimate))

    # Independent recomputation from the documented formula:
    # psi-tilde(v) = 1{v <= t} - F0(t), then var = mean(psi^2).
    n = len(np.asarray(x, dtype=float).ravel())
    F_t = sum(1 for v in np.asarray(x, dtype=float).ravel() if v <= t) / n
    infl = [(1.0 if v <= t else 0.0) - F_t for v in np.asarray(x, dtype=float).ravel()]
    expected_var = sum(v * v for v in infl) / n
    expected_mean_gap = abs(sum(infl) / n)

    assert abs(estimate - expected_var) < 1e-12
    assert abs(float(result["mean_zero_gap"]) - expected_mean_gap) < 1e-12
    assert result["matches_bernoulli_var"] is True
    assert result["method"] == "efficient influence function (GvdV 2017 sec. 12.3.1)"


def test_gh_c12_5_edge():
    """Test edge cases."""
    result = ghosal_eff_infl_fn(np.array([42.0]), 42.0)
    # With a single observation equal to t, F_t = 1, infl = 1 - 1 = 0,
    # so the empirical efficient-influence-function variance is exactly 0.
    assert abs(float(result["estimate"])) < 1e-12
    assert abs(float(result["mean_zero_gap"])) < 1e-12
    # Bernoulli variance at F_t = 1 is 1*0 = 0, so they should match.
    assert result["matches_bernoulli_var"] is True
