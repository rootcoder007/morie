"""Tests for gh_emp_bayes.ghosal_empirical_bayes_np."""

from morie.fn import _array_core as np

from morie.fn.gh_emp_bayes import ghosal_empirical_bayes_np


def test_gh_emp_bayes_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_empirical_bayes_np(x)
    assert "alpha_hat" in result
    assert np.all(np.isfinite(np.asarray(result["alpha_hat"], dtype=float)))
    # Independent check: k = number of distinct values, n = sample size
    n = int(x.size)
    k = int(np.unique(x).size)
    assert result["n"] == n
    assert result["n_clusters"] == k
    # understates_uncertainty is documented as always True for EB
    assert result["understates_uncertainty"] is True
    # alpha_hat should come from the supplied alpha_grid
    ag = np.asarray(result["alpha_grid"], dtype=float)
    assert np.any(np.isclose(ag, result["alpha_hat"]))
    # log_marginal should be maximised at the chosen alpha_hat
    lm = np.asarray(result["log_marginal"], dtype=float)
    j = int(np.argmax(lm))
    assert np.isclose(ag[j], result["alpha_hat"])


def test_gh_emp_bayes_edge():
    """Test edge cases."""
    x = np.array([42.0, 42.0])
    result = ghosal_empirical_bayes_np(x)
    # All observations tie -> single cluster
    assert result["n_clusters"] == 1
    assert result["n"] == 2
    assert result["alpha_hat"] > 0
    assert np.all(np.isfinite(np.asarray(result["log_marginal"], dtype=float)))
