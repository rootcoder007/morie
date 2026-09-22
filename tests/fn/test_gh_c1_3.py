"""Tests for gh_c1_3.ghosal_prior_posterior_update."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c1_3 import ghosal_prior_posterior_update


def test_gh_c1_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_prior_posterior_update(x)
    assert "estimate" in result
    assert "posterior" in result
    assert "sequential_batch_gap" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    posterior = result["posterior"]
    assert len(posterior) == len(x)
    assert abs(sum(posterior) - 1.0) < 1e-12
    assert result["sequential_batch_gap"] < 1e-12

    # Independent computation of the posterior for the default data
    # log_lik_one(t, d) = -0.5 * (d - t)**2, log_prior(t) = -0.5 * t * t,
    # data = [0.8, 1.2, 1.0]; log w_t = sum_d log_lik_one(t, d) + log_prior(t).
    data = [0.8, 1.2, 1.0]
    x_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    lw = [
        sum(-0.5 * (d - t) ** 2 for d in data) + (-0.5 * t * t)
        for t in x_list
    ]
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    tot = sum(w)
    expected_post = [v / tot for v in w]
    expected_est = sum(t * p for t, p in zip(x_list, expected_post))
    for got, exp in zip(posterior, expected_post):
        assert abs(got - exp) < 1e-12
    assert abs(result["estimate"] - expected_est) < 1e-12


def test_gh_c1_3_edge():
    """Test edge case: single-point support for theta."""
    x = np.array([42.0])
    result = ghosal_prior_posterior_update(x)
    # 'n' is not a key the function exposes; check returned keys directly.
    assert "estimate" in result
    assert "posterior" in result
    assert "sequential_batch_gap" in result
    assert "method" in result

    posterior = result["posterior"]
    assert len(posterior) == 1
    # All posterior mass concentrates on the single grid point.
    assert abs(posterior[0] - 1.0) < 1e-12
    # With a single theta, the posterior mean equals that theta value.
    assert abs(result["estimate"] - 42.0) < 1e-12
    # Sequential vs. batch updating must agree exactly.
    assert result["sequential_batch_gap"] < 1e-12
