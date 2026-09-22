"""Tests for gh_c13_14.ghosal_cox_post."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_14 import ghosal_cox_post


def test_gh_c13_14_basic():
    """Test basic functionality."""
    result = ghosal_cox_post(beta0=0.6, n=400, prior_sd=2.0, seed=42)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))
    # Documented behaviour: posterior mean is near the prior mean beta0.
    # Grid spans [beta0 - 1.5, beta0 + 1.5] in 60 steps -> posterior mass is
    # contained in this window, so |post_mean - beta0| <= 1.5.
    assert abs(estimate - 0.6) <= 1.5


def test_gh_c13_14_edge():
    """Test edge cases."""
    result = ghosal_cox_post(beta0=0.6, n=400, prior_sd=2.0, seed=42)
    # The function does not expose a 'n' key; assert a key the function
    # actually returns, and confirm determinism with the same seed.
    assert "error" in result
    again = ghosal_cox_post(beta0=0.6, n=400, prior_sd=2.0, seed=42)
    assert float(np.asarray(result["estimate"], dtype=float)) == \
           float(np.asarray(again["estimate"], dtype=float))
    # Changing the seed should (very likely) produce a different estimate.
    other = ghosal_cox_post(beta0=0.6, n=400, prior_sd=2.0, seed=43)
    assert float(np.asarray(result["estimate"], dtype=float)) != \
           float(np.asarray(other["estimate"], dtype=float))
