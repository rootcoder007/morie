"""spglmm -- conditional GLMM specification, Schabenberger Sec. 6.3.4."""

from morie.fn import _array_core as np
import pytest

from morie.fn.spglmm import schabenberger_glmm_conditional


def _design(n=25, seed=11):
    rs = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    return X, np.array([0.4, 0.8]), rs.normal(0, 0.7, n)


def test_conditional_mean_is_the_inverse_link_of_the_linear_predictor():
    """eq (6.73): g[mu(s)] = x(s)'beta + S(s)."""
    X, beta, S = _design()
    r = schabenberger_glmm_conditional(X, beta, S, family="poisson")
    assert np.allclose(r["conditional_mean"], np.exp(X @ beta + S))


def test_conditional_variance_is_sigma2_times_v_of_mu():
    """eq (6.74), with v(mu) = mu for the Poisson."""
    X, beta, S = _design()
    r = schabenberger_glmm_conditional(X, beta, S, sigma2=2.0, family="poisson")
    assert np.allclose(r["conditional_variance"], 2.0 * r["conditional_mean"])


def test_marginal_mean_is_not_the_inverse_link_at_x_beta():
    """The central warning of Sec. 6.3.4."""
    X, beta, S = _design()
    r = schabenberger_glmm_conditional(X, beta, S, family="poisson")
    assert not np.allclose(r["marginal_mean"], r["naive_marginal_mean"])
    # and the discrepancy is exactly exp(sigma_S^2 / 2)
    assert r["marginal_ratio"] == pytest.approx(np.exp(r["sigma2_S"] / 2))
    assert np.allclose(r["marginal_mean"] / r["naive_marginal_mean"],
                       r["marginal_ratio"])


def test_the_gap_grows_with_the_latent_variance():
    """A bigger latent field makes the naive value worse, not merely noisy."""
    X, beta, S = _design()
    small = schabenberger_glmm_conditional(X, beta, 0.2 * S, family="poisson")
    large = schabenberger_glmm_conditional(X, beta, 3.0 * S, family="poisson")
    assert large["marginal_ratio"] > small["marginal_ratio"] > 1.0


def test_latent_field_induces_overdispersion():
    """Example 6.6: Var[Z] > E[Z] even at sigma^2 = 1."""
    X, beta, S = _design()
    r = schabenberger_glmm_conditional(X, beta, S, sigma2=1.0, family="poisson")
    assert np.all(r["marginal_variance"] > r["marginal_mean"])


def test_marginal_covariance_matches_the_variance_on_the_diagonal():
    """The book's Cov expression must reduce to Var at i = j."""
    X, beta, S = _design()
    n = X.shape[0]
    r = schabenberger_glmm_conditional(X, beta, S, family="poisson",
                                       correlation=np.eye(n))
    # Var[Z] = E[Var(Z|S)] + Var(E[Z|S]); the second part is the diagonal of
    # the covariance, and at sigma^2 = 1 the first is m exp(sigma_S^2/2),
    # which is the marginal mean itself.
    assert np.allclose(r["marginal_variance"],
                       np.diag(r["marginal_covariance"]) + r["marginal_mean"])


def test_non_log_link_says_there_is_no_closed_form():
    X, beta, S = _design()
    r = schabenberger_glmm_conditional(X, beta, S, family="binomial")
    assert "no closed" in r["marginal_note"]
    assert "marginal_mean" not in r
