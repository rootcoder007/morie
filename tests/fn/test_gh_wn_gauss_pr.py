"""Tests for gh_wn_gauss_pr.ghosal_white_noise_gauss_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_wn_gauss_pr import ghosal_white_noise_gauss_prior


def test_gh_wn_gauss_pr_basic():
    """Test basic functionality against the closed-form GP posterior formula.

    Per the docstring (sec. 9.5.4):
        theta | Y ~ N(C (C + I/n)^{-1} Y, (C^{-1} + n I)^{-1})
    where C is diagonal with C_ii = prior_sd_i**2.
    """
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = 3.0
    prior_sd = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

    result = ghosal_white_noise_gauss_prior(Y, n, prior_sd)

    # The function is documented to return a key named "estimate".
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    # Also check the additional documented payload keys are present and finite.
    posterior_mean = np.asarray(result["posterior_mean"], dtype=float)
    posterior_var = np.asarray(result["posterior_var"], dtype=float)
    assert posterior_mean.shape == Y.shape
    assert posterior_var.shape == Y.shape
    assert np.all(np.isfinite(posterior_mean))
    assert np.all(np.isfinite(posterior_var))

    # Compute expected posterior statistics independently from the formula.
    Y_flat = np.asarray(Y, dtype=float).reshape(-1)
    c = np.asarray(prior_sd, dtype=float).reshape(-1) ** 2
    n_f = float(n)
    expected_mean = c / (c + 1.0 / n_f) * Y_flat
    expected_var = 1.0 / (1.0 / c + n_f)

    # estimate is documented to be means[0]; compare against the first coordinate
    # of the formula-derived posterior mean.
    assert estimate == expected_mean[0]

    # Full posterior mean/var must match the formula applied coordinatewise.
    assert np.allclose(posterior_mean, expected_mean)
    assert np.allclose(posterior_var, expected_var)


def test_gh_wn_gauss_pr_edge():
    """Test edge cases with a single observation and a constant prior."""
    Y = np.array([42.0])
    n = 1.0
    prior_sd = np.array([1.0])

    result = ghosal_white_noise_gauss_prior(Y, n, prior_sd)

    # "estimate" is the documented return key; the function does NOT return "n".
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))

    # Independently computed from the formula with c = prior_sd**2 = 1
    # and n = 1:  mean = 1 / (1 + 1) * y = y / 2;  var = 1 / (1 + 1) = 1/2.
    c = float(prior_sd[0]) ** 2
    expected_mean = c / (c + 1.0 / float(n)) * float(Y[0])
    expected_var = 1.0 / (1.0 / c + float(n))
    assert estimate == expected_mean
    assert np.allclose(np.asarray(result["posterior_mean"], dtype=float), expected_mean)
    assert np.allclose(np.asarray(result["posterior_var"], dtype=float), expected_var)
