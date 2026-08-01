"""sppql -- pseudo-likelihood for spatial GLMMs, Schabenberger Sec. 6.3.5."""

import numpy as np
import pytest

from morie.fn import _schab_glmm as gm
from morie.fn.sppql import schabenberger_pql


def _data(n=30, seed=7):
    rs = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    beta = np.array([0.5, 0.7])
    d = np.abs(np.subtract.outer(np.linspace(0, 8, n), np.linspace(0, 8, n)))
    Sigma_S = 0.4 * np.exp(-d / 2.5)
    S = np.linalg.cholesky(Sigma_S + 1e-10 * np.eye(n)) @ rs.normal(0, 1, n)
    y = np.array([float(rs.poisson(m)) for m in np.exp(X @ beta + S)])
    return y, X, Sigma_S, beta


def test_fit_converges_and_reports_iterations():
    y, X, Sigma_S, _ = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson")
    assert r["converged"] and r["n_iter"] >= 1
    assert "warning" not in r


def test_pql_and_pl_agree_as_the_text_says_they_must():
    """Sec. 6.3.5.3: the two objectives differ only by a constant."""
    y, X, Sigma_S, _ = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson", check_score=True)
    assert r["pql_pl_equivalent"]
    assert r["score_beta_max"] < 1e-6
    assert r["score_S_max"] < 1e-6
    assert "score_warning" not in r


def test_pseudo_data_matches_equation_6_78():
    y, X, Sigma_S, _ = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson")
    mu = r["mu"]
    assert np.allclose(r["pseudo_data"], np.log(mu) + (y - mu) / mu)


def test_standard_errors_come_from_the_gls_information():
    """Var(beta_hat) = (X' Sigma_nu^-1 X)^-1."""
    y, X, Sigma_S, _ = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson")
    Sinv = np.linalg.inv(r["Sigma_nu"])
    assert np.allclose(r["cov_beta"], np.linalg.inv(X.T @ Sinv @ X))
    assert np.allclose(r["se_beta"], np.sqrt(np.diag(r["cov_beta"])))


def test_recovers_the_generating_coefficients():
    y, X, Sigma_S, beta = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson")
    assert np.all(np.abs(r["beta"] - beta) < 3 * r["se_beta"])


def test_marginal_and_conditional_specifications_are_distinguished():
    """The text's choice: dependence in R, or in Sigma_S."""
    y, X, Sigma_S, _ = _data()
    n = y.size
    d = np.abs(np.subtract.outer(np.linspace(0, 8, n), np.linspace(0, 8, n)))
    cond = schabenberger_pql(y, X, Sigma_S, family="poisson")
    marg = schabenberger_pql(y, X, 1e-8 * np.eye(n), family="poisson",
                             R=np.exp(-d / 2.5))
    assert cond["specification"] == "conditional"
    assert marg["specification"] == "marginal"


def test_reml_objective_is_reported_and_finite():
    y, X, Sigma_S, _ = _data()
    r = schabenberger_pql(y, X, Sigma_S, family="poisson")
    assert np.isfinite(r["reml"])


def test_mismatched_shapes_rejected():
    y, X, Sigma_S, _ = _data()
    with pytest.raises(ValueError, match="sample size"):
        schabenberger_pql(y[:-1], X, Sigma_S, family="poisson")
