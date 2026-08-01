"""spglmk -- spatial prediction in GLMs, Schabenberger Sec. 6.3.6."""

import numpy as np
import pytest

from morie.fn.spglmk import schabenberger_glm_kriging


def _setup(n=20, seed=5):
    rs = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    t = np.linspace(0, 8, n)
    d = np.abs(np.subtract.outer(t, t))
    Sigma = 0.5 * np.exp(-d / 2.0) + 0.05 * np.eye(n)
    nu = rs.normal(0.5, 0.4, n)
    s0 = 0.5 * np.exp(-np.abs(t - 4.0) / 2.0)
    return nu, Sigma, s0, X, np.array([1.0, 0.1])


def test_the_two_predictors_are_different_quantities():
    """eq (6.87) and eq (6.90) do not coincide."""
    nu, S, s0, X, x0 = _setup()
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=2.0, link_kind="log")
    assert r["prediction"] != r["inverse_link_prediction"]


def test_predictor_follows_equation_6_90():
    nu, S, s0, X, x0 = _setup()
    mu0 = 2.0
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=mu0, link_kind="log")
    nu0 = r["pseudo_scale_prediction"]
    assert r["prediction"] == pytest.approx(mu0 + mu0 * (nu0 - np.log(mu0)))


def test_mspe_follows_equation_6_91():
    """sigma^2_Z = (dmu/deta)^2 sigma^2_nu; for the log link dmu/deta = mu."""
    nu, S, s0, X, x0 = _setup()
    mu0 = 2.0
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=mu0, link_kind="log")
    assert r["mspe"] == pytest.approx(mu0**2 * r["pseudo_scale_mspe"])
    assert r["prediction_error"] == pytest.approx(np.sqrt(r["mspe"]))


def test_the_result_says_which_predictor_the_mspe_belongs_to():
    """The book warns (6.88) is not the MSPE of (6.87)."""
    nu, S, s0, X, x0 = _setup()
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=2.0, link_kind="log")
    assert "6.90" in r["mspe_is_for"] and "6.87" in r["mspe_is_for"]


def test_kriging_variance_is_non_negative():
    nu, S, s0, X, x0 = _setup()
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=2.0, link_kind="log")
    assert r["pseudo_scale_mspe"] >= 0.0


def test_identity_link_leaves_the_prediction_on_its_own_scale():
    nu, S, s0, X, x0 = _setup()
    r = schabenberger_glm_kriging(nu, S, s0, X, x0, mu0=2.0,
                                  link_kind="identity")
    assert r["prediction"] == pytest.approx(r["pseudo_scale_prediction"])
    assert r["mspe"] == pytest.approx(r["pseudo_scale_mspe"])


def test_shape_mismatches_rejected():
    nu, S, s0, X, x0 = _setup()
    with pytest.raises(ValueError, match="agree on n"):
        schabenberger_glm_kriging(nu[:-1], S, s0, X, x0, mu0=2.0)
    with pytest.raises(ValueError, match="one entry per column"):
        schabenberger_glm_kriging(nu, S, s0, X, np.array([1.0]), mu0=2.0)
