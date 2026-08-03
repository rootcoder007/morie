"""Known-answer tests for MVSML chapter 6 (Bayesian genomic
linear regression), eq. (6.1)-(6.7).

The strongest checks are the equivalences the book itself states:
GBLUP is the BRR run on the Cholesky factor of G (p.177), BayesC
collapses to the BRR when pi_0 = 1 (p.180), and BayesB collapses to
BayesA when pi = 1 (p.183).
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm042 import mvsml_bayesian_regression_eq_6_1
from morie.fn.msm043 import mvsml_bayesian_regression_eq_6_2
from morie.fn.msm046 import mvsml_bayesian_regression_eq_6_3
from morie.fn.msm049 import mvsml_bayesian_regression_eq_6_4
from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5
from morie.fn.msm061 import mvsml_bayesian_regression_eq_6_6
from morie.fn.msm063 import mvsml_bayesian_regression_eq_6_7


def _sim(n=40, p=4, seed=7):
    rng = gp.np.random.default_rng(seed)
    X = [[float(rng.normal(0, 1)) for _ in range(p)]
         for _ in range(n)]
    truth = [2.0, 0.0, -1.5, 0.0][:p]
    y = [sum(a * b for a, b in zip(row, truth))
         + 0.3 * float(rng.normal(0, 1)) for row in X]
    return X, y, truth


def test_eq_6_1_posterior_matches_the_ols_algebra():
    X, y, truth = _sim()
    r = mvsml_bayesian_regression_eq_6_1(X, y)
    f = gp.ols_fit(X, y, add_intercept=True)
    # under the reference prior the posterior mean of beta is the OLS
    # estimator and its scale is sigma2 (X'X)^-1 (p.172)
    for a, b in zip(r["posterior_mean_beta"], f["beta"]):
        assert abs(a - b) < 1e-12
    for a, b in zip(r["posterior_sd_beta"], f["se_beta"]):
        assert abs(a - b) < 1e-12
    # the inverse-gamma mean exceeds the plug-in estimate
    assert r["posterior_mean_sigma2"] > r["sigma2_hat"]
    for got, want in zip(r["posterior_mean_beta"][1:], truth):
        assert abs(got - want) < 0.2


def test_eq_6_2_prior_is_improper_and_scale_free():
    r = mvsml_bayesian_regression_eq_6_2(2.0)
    assert abs(r["density"] - 0.25) < 1e-12
    assert r["proper"] is False
    assert abs(r["log_density"] + 2.0 * math.log(2.0)) < 1e-12
    # proportional to sigma^-2: doubling sigma2 quarters the density
    assert abs(mvsml_bayesian_regression_eq_6_2(4.0)["density"]
               - r["density"] / 4.0) < 1e-12


def test_eq_6_3_brr_recovers_the_signal_and_shrinks_the_noise():
    X, y, truth = _sim()
    r = mvsml_bayesian_regression_eq_6_3(y, X, n_iter=1500,
                                         burn_in=400)
    assert abs(r["beta"][0] - 2.0) < 0.25
    assert abs(r["beta"][2] + 1.5) < 0.25
    assert abs(r["beta"][1]) < 0.2 and abs(r["beta"][3]) < 0.2
    assert r["sigma2"] > 0 and r["sigma2_beta"] > 0
    assert r["n_kept"] == 1100


def test_brr_hyperparameters_match_the_bglr_defaults():
    X, y, _ = _sim()
    hp = gp.brr_hyperparameters(y, R2=0.5, nu=5.0, nu_beta=5.0)
    # p.175: S = Var(Y)(1-R2)(nu+2), S_beta = Var(Y) R2 (nu_beta+2)
    assert abs(hp["S"] - hp["var_y"] * 0.5 * 7.0) < 1e-12
    assert abs(hp["S_beta"] - hp["var_y"] * 0.5 * 7.0) < 1e-12
    # BayesC divides the slab scale by the sum of column variances
    hp2 = gp.brr_hyperparameters(y, sum_var_x=4.0)
    assert abs(hp2["S_beta"] - hp["S_beta"] / 4.0) < 1e-12


def test_eq_6_4_gblup_is_the_brr_on_the_cholesky_factor():
    # book p.177: "The GBLUP can be equivalently expressed and
    # consequently fitted with the BRR model by making the design
    # matrix equal to the lower triangular factor of the Cholesky
    # decomposition of G"
    rng = gp.np.random.default_rng(11)
    n = 12
    A = [[float(rng.normal(0, 1)) for _ in range(6)]
         for _ in range(n)]
    G = gp.grm_vanraden_method3(A)
    G = [[G[i][j] + (0.3 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    y = [5.0 + float(rng.normal(0, 1)) for _ in range(n)]
    L = gp.cholesky_lower(G)
    direct = gp.bayes_ridge_gibbs(y, L, n_iter=600, burn_in=200,
                                  seed=3)
    r = mvsml_bayesian_regression_eq_6_4(y, G, n_iter=600,
                                         burn_in=200, seed=3)
    assert abs(r["mu"] - direct["mu"]) < 1e-12
    for a, b in zip(r["g"], gp._mv(L, direct["beta"])):
        assert abs(a - b) < 1e-12
    # the Cholesky factor really does reproduce G
    prod = gp._mm(L, gp._t(L))
    assert max(abs(prod[i][j] - G[i][j])
               for i in range(n) for j in range(n)) < 1e-9


def test_eq_6_5_covariance_of_the_predictor():
    # K_L = Z G Z' (p.177)
    Z = [[1, 0], [1, 0], [0, 1], [0, 1]]
    G = [[1.0, 0.4], [0.4, 1.0]]
    r = mvsml_bayesian_regression_eq_6_5(
        [5.0, 5.2, 6.0, 6.1], Z, G, n_iter=300, burn_in=100)
    K = r["K_L"]
    assert abs(K[0][0] - 1.0) < 1e-12      # both rows are line 1
    assert abs(K[0][1] - 1.0) < 1e-12
    assert abs(K[0][2] - 0.4) < 1e-12      # line 1 vs line 2
    assert len(K) == 4


def test_eq_6_6_extended_predictor_block_layout():
    n = 6
    X_E = [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3
    Xm = [[0.5, -0.5, 1.0]] * 6
    X_EM = [[0.1, 0.2]] * 6
    r = mvsml_bayesian_regression_eq_6_6(n, X_E=X_E, X=Xm,
                                         X_EM=X_EM)
    assert r["widths"] == {"intercept": 1, "environments": 2,
                           "markers": 3, "env_x_marker": 2}
    assert r["estimate"] == 8.0
    assert r["design"][0][0] == 1.0
    # blocks appear in the order of eq. (6.6)
    assert r["design"][0][1:3] == [1.0, 0.0]
    assert r["design"][0][3:6] == [0.5, -0.5, 1.0]


def test_eq_6_7_gxe_covariances():
    Z_L = [[1, 0], [0, 1], [1, 0], [0, 1]]
    G = [[1.0, 0.3], [0.3, 1.0]]
    I_env = [[1.0, 0.0], [0.0, 1.0]]
    Z_LE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    r = mvsml_bayesian_regression_eq_6_7(Z_L, G, Z_LE=Z_LE,
                                         I_env=I_env)
    assert abs(r["K_L"][0][0] - 1.0) < 1e-12
    assert abs(r["K_L"][0][1] - 0.3) < 1e-12
    # K_LE keeps environments independent: the (1,3) block is zero
    K = r["K_LE"]
    assert abs(K[0][0] - 1.0) < 1e-12
    assert abs(K[0][2]) < 1e-12


def test_bayes_c_reduces_to_the_brr_when_pi0_is_one():
    # book p.180: "BayesC is reduced to BRR when pi_p0 = 1"
    X, y, _ = _sim(n=30, p=3, seed=5)
    bc = gp.bayes_c_gibbs(y, X, n_iter=700, burn_in=200, pi0=1.0,
                          seed=9)
    brr = gp.bayes_ridge_gibbs(y, X, n_iter=700, burn_in=200,
                               seed=9)
    assert all(v > 0.999 for v in bc["inclusion_prob"])
    for a, b in zip(bc["beta"], brr["beta"]):
        assert abs(a - b) < 0.15


def test_bayes_b_reduces_to_bayes_a_when_pi_is_one():
    # book p.183: "if pi = 1, this model is reduced to BayesA"
    X, y, _ = _sim(n=30, p=3, seed=5)
    bb = gp.bayes_b_gibbs(y, X, n_iter=700, burn_in=200, pi0=1.0,
                          seed=13)
    ba = gp.bayes_a_gibbs(y, X, n_iter=700, burn_in=200, seed=13)
    assert all(v > 0.999 for v in bb["inclusion_prob"])
    for a, b in zip(bb["beta"], ba["beta"]):
        assert abs(a - b) < 0.15


def test_bayes_c_selects_the_nonzero_covariates():
    X, y, truth = _sim(n=60, p=4, seed=21)
    r = gp.bayes_c_gibbs(y, X, n_iter=900, burn_in=300, pi0=0.5,
                         seed=4)
    incl = r["inclusion_prob"]
    assert incl[0] > 0.9 and incl[2] > 0.9        # true signals
    assert incl[1] < 0.5 and incl[3] < 0.5        # true nulls


def test_bayes_lasso_shrinks_more_than_the_brr():
    X, y, _ = _sim(n=40, p=4, seed=7)
    bl = gp.bayes_lasso_gibbs(y, X, n_iter=900, burn_in=300,
                              lam2=1.0, seed=6)
    brr = gp.bayes_ridge_gibbs(y, X, n_iter=900, burn_in=300,
                               seed=6)
    # the double-exponential prior puts more mass at zero (p.184)
    assert abs(bl["beta"][0]) < abs(brr["beta"][0])
    assert bl["beta"][0] > 0.5          # but the signal survives


def test_scaled_inverse_chisq_has_the_right_mean():
    rng = gp.np.random.default_rng(2)
    nu, S = 20.0, 40.0
    draws = [gp.scaled_inv_chisq(rng, nu, S) for _ in range(4000)]
    # E[S/chi2_nu] = S/(nu - 2)
    assert abs(sum(draws) / len(draws) - S / (nu - 2.0)) < 0.25
