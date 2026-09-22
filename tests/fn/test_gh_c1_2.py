"""Tests for gh_c1_2.ghosal_absolute_continuity."""

from morie.fn import _array_core as np

from morie.fn.gh_c1_2 import ghosal_absolute_continuity


def test_gh_c1_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_absolute_continuity(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c1_2_edge():
    """Test edge case: single-point grid yields a sensible single-point posterior."""
    theta = np.array([42.0])
    result = ghosal_absolute_continuity(theta)

    # Posterior should have the same length as the input grid.
    posterior = np.asarray(result["posterior"], dtype=float)
    assert posterior.shape[0] == theta.shape[0]

    # On a single-point grid, the posterior puts all mass at that single theta.
    assert np.all(np.isfinite(posterior))
    assert float(posterior[0]) == 1.0

    # The posterior mean estimate must match the (sole) grid point.
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    assert float(estimate) == float(np.asarray(theta, dtype=float).reshape(-1)[0])

    # Method label and log_marginal are documented return values; marginal must be finite.
    assert result["method"] == "dominated posterior, Radon-Nikodym (GvdV 2017 sec. 1.3.1)"
    assert np.all(np.isfinite(np.asarray(result["log_marginal"], dtype=float)))


def test_gh_c1_2_log_marginal_matches_formula():
    """Independently verify the log_marginal against the documented log-sum-exp formula."""
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    def log_lik(t):
        return -0.5 * (1.0 - t) ** 2

    def log_prior(t):
        return -0.5 * t * t

    import math

    theta = np.asarray(x, dtype=float).reshape(-1)
    lw = [log_lik(float(t)) + log_prior(float(t)) for t in theta]
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    tot = sum(w)
    expected_log_marginal = math.log(tot / len(theta)) + mx
    expected_posterior = [v / tot for v in w]

    result = ghosal_absolute_continuity(x, log_lik=log_lik, log_prior=log_prior)

    assert np.allclose(
        np.asarray(result["log_marginal"], dtype=float),
        np.asarray(expected_log_marginal, dtype=float),
    )
    posterior = np.asarray(result["posterior"], dtype=float)
    assert posterior.shape == np.asarray(expected_posterior, dtype=float).shape
    assert np.allclose(posterior, np.asarray(expected_posterior, dtype=float))
