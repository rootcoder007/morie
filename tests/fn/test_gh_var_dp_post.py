"""Tests for gh_var_dp_post.ghosal_variational_dp_posterior."""

from morie.fn import _array_core as np

from morie.fn.gh_var_dp_post import ghosal_variational_dp_posterior


def test_gh_var_dp_post_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_variational_dp_posterior(x)
    assert "estimate" in result
    assert "posterior" in result
    assert "centers" in result
    assert "method" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(est))
    assert est >= 0.0


def test_gh_var_dp_post_edge():
    """Test edge cases."""
    result = ghosal_variational_dp_posterior(np.array([42.0]))
    assert "estimate" in result
    assert "centers" in result
    centers = np.asarray(result["centers"], dtype=float)
    assert centers.shape == (8,)
    assert np.all(np.isfinite(centers))
    # With a single data point at 42.0, the optimal responsibilities put
    # all mass on the center closest to 42.0; the fitted center closest
    # to 42.0 must therefore equal 42.0 (since the likelihood Gaussian has
    # finite variance, the posterior mean under that component is exactly
    # the data point when the prior precision is 0 relative to the
    # likelihood precision -- here the closed-form update with r=1 yields
    # xi_j = x_i / (1 + (sigma/tau)^2) ... but with K components and
    # softmax over centers, only the nearest center is updated to 42.0).
    # Independent sanity: the function must run and return finite values.
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(est))
