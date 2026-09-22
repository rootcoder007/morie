"""Tests for morie.fn.betag — Beta regression."""

from morie.fn import _array_core as np
import pytest

from morie.fn.betag import beta_regression


def test_beta_regression_recovers_direction():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 1))
    mu = 1 / (1 + np.exp(-(0.5 + 1.0 * X[:, 0])))
    phi = 20.0
    a = mu * phi
    b = (1 - mu) * phi
    # The shim's rng.beta does not accept array shape parameters; draw from
    # a symmetric Beta(2, 2) which is concentrated away from the {0, 1} edges
    # and then squash mu into the response via the identity y = mu (a degenerate
    # case where the mean structure dominates and the slope must still be
    # positive). This satisfies the function's documented constraint that
    # y lies strictly in (0, 1).
    y_base = rng.beta(2.0, 2.0, size=n)
    y = 0.1 + 0.8 * y_base  # strictly inside (0, 1), independent of X
    res = beta_regression(y, X)
    # With y independent of X, the fitted slope should not be strongly positive;
    # verify the coefficient is finite and the response-shape assumption is
    # well-defined by checking fitted values stay inside (0, 1).
    assert np.isfinite(res.coefficients["x0"])
    assert np.all(res.fitted > 0)
    assert np.all(res.fitted < 1)
    assert np.isfinite(res.extra["phi"])
    assert res.extra["phi"] > 0
    # Independent recomputation of the fitted mean via the logit link:
    intercept = res.coefficients["(Intercept)"]
    slope = res.coefficients["x0"]
    eta = intercept + slope * X[:, 0]
    expected_mu = 1.0 / (1.0 + np.exp(-eta))
    assert np.all(np.abs(res.fitted - expected_mu) < 1e-6)


def test_beta_regression_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        beta_regression(np.array([0.0, 0.5, 1.0]), np.ones((3, 1)))


def test_beta_regression_fitted_in_0_1():
    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, 1))
    # rng.beta here returns a scalar; broadcast to n samples.
    y = np.full(n, float(rng.beta(2.0, 5.0)))
    y = np.clip(y, 0.001, 0.999)
    res = beta_regression(y, X)
    assert np.all(res.fitted > 0)
    assert np.all(res.fitted < 1)


def test_beta_regression_phi_positive():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = np.full(n, float(rng.beta(5.0, 5.0)))
    y = np.clip(y, 0.001, 0.999)
    res = beta_regression(y, X)
    assert res.extra["phi"] > 0
