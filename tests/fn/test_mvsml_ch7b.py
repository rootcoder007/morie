"""Known-answer tests for MVSML chapter 7, eq. (7.3)-(7.11)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm092 import mvsml_bayesian_regression_pt2_eq_7_3
from morie.fn.msm095 import mvsml_bayesian_regression_pt2_eq_7_4
from morie.fn.msm098 import mvsml_bayesian_regression_pt2_eq_7_5
from morie.fn.msm106 import mvsml_bayesian_regression_pt2_eq_7_6
from morie.fn.msm109 import mvsml_bayesian_regression_pt2_eq_7_7
from morie.fn.msm110 import mvsml_bayesian_regression_pt2_eq_7_8
from morie.fn.msm112 import mvsml_bayesian_regression_pt2_eq_7_9
from morie.fn.msm115 import mvsml_bayesian_regression_pt2_eq_7_10
from morie.fn.msm122 import mvsml_bayesian_regression_pt2_eq_7_11


def test_eq_7_3_stacks_the_three_predictor_blocks():
    n = 4
    X_E = [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2
    Xm = [[0.5, -0.5, 1.0]] * n
    X_EM = [[0.1, 0.2]] * n
    r = mvsml_bayesian_regression_pt2_eq_7_3(n, X_E=X_E, X=Xm,
                                             X_EM=X_EM)
    assert r["widths"] == {"environments": 2, "markers": 3,
                           "env_x_marker": 2}
    assert r["estimate"] == 7.0
    assert r["design"][0][:2] == [1.0, 0.0]


def test_eq_7_5_uses_the_cholesky_factor_for_the_genetic_block():
    # Table 7.6 p.233: X = [X_E, Z_L L_g] with G = L_g L_g'
    G = [[1.0, 0.3], [0.3, 1.0]]
    L = gp.cholesky_lower(G)
    Z = [[1.0, 0.0], [0.0, 1.0]]
    r = mvsml_bayesian_regression_pt2_eq_7_5(2, [[1.0], [1.0]], Z,
                                             L_g=L)
    assert r["widths"] == {"environments": 1, "genetic": 2}
    for i in range(2):
        for j in range(2):
            assert abs(r["design"][i][1 + j] - L[i][j]) < 1e-12


def test_eq_7_4_ordinal_gblup_separates_categories():
    rng = gp.np.random.default_rng(5)
    n = 16
    A = [[float(rng.normal(0, 1)) for _ in range(24)]
         for _ in range(n)]
    G = gp.grm_vanraden_method3(A)
    G = [[G[i][j] + (0.5 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    y = [1] * (n // 2) + [3] * (n // 2)
    r = mvsml_bayesian_regression_pt2_eq_7_4(y, G, n_iter=400,
                                             burn_in=150)
    lo = sum(r["b"][:n // 2]) / (n // 2)
    hi = sum(r["b"][n // 2:]) / (n // 2)
    assert hi < lo                      # l_i is centred at -b_i
    assert r["gamma"][0] < r["gamma"][1]


def test_eq_7_6_probabilities_normalize_and_use_the_baseline():
    X = [[1.0, 0.0], [0.0, 1.0]]
    r = mvsml_bayesian_regression_pt2_eq_7_6(X, [0.5, -0.5],
                                             [[1.0, 0.0],
                                              [0.0, 1.0]])
    P = r["probabilities"]
    assert r["n_categories"] == 3       # two free classes + baseline
    for row in P:
        assert abs(sum(row) - 1.0) < 1e-12
    # hand value: class 1 for x = (1,0) is exp(1.5)/(exp(1.5)
    # + exp(-0.5) + exp(0))
    d = math.exp(1.5) + math.exp(-0.5) + 1.0
    assert abs(P[0][0] - math.exp(1.5) / d) < 1e-12
    assert abs(P[0][2] - 1.0 / d) < 1e-12


def test_eq_7_8_loglik_matches_the_log_of_the_probabilities():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    y = [0, 1, 2]
    b0, B = [0.5, -0.5], [[1.0, 0.0], [0.0, 1.0]]
    r = mvsml_bayesian_regression_pt2_eq_7_8(X, y, b0, B)
    P = gp.multinomial_probabilities(X, b0, B)
    hand = sum(math.log(P[i][y[i]]) for i in range(3))
    assert abs(r["loglik"] - hand) < 1e-12
    assert r["loglik"] < 0.0


def test_eq_7_7_and_7_10_penalties():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    y = [0, 1, 2]
    b0, B = [0.5, -0.5], [[1.0, -2.0], [0.0, 1.0]]
    ridge = mvsml_bayesian_regression_pt2_eq_7_7(X, y, b0, B,
                                                 lam=2.0)
    lasso = mvsml_bayesian_regression_pt2_eq_7_10(X, y, b0, B,
                                                  lam=2.0)
    # ridge: lambda * sum beta^2 = 2 * (1 + 4 + 0 + 1) = 12
    assert abs(ridge["penalty"] - 12.0) < 1e-12
    # lasso: lambda * sum |beta| = 2 * (1 + 2 + 0 + 1) = 8
    assert abs(lasso["penalty"] - 8.0) < 1e-12
    assert abs(ridge["loglik"] - lasso["loglik"]) < 1e-12
    assert ridge["estimate"] < ridge["loglik"]      # penalty applied


def test_eq_7_9_block_update_improves_the_objective():
    rng = gp.np.random.default_rng(3)
    n = 60
    X = [[float(rng.normal(0, 1))] for _ in range(n)]
    y = [0 if row[0] < -0.3 else (1 if row[0] < 0.4 else 2)
         for row in X]
    b0, B = [0.0, 0.0], [[0.0], [0.0]]
    before = gp.penalized_multinomial_loglik(X, y, b0, B, 0.1)
    upd = mvsml_bayesian_regression_pt2_eq_7_9(X, y, b0, B,
                                               lam=0.1, cls=0)
    b0n = [upd["beta0"], b0[1]]
    Bn = [upd["beta"], B[1]]
    after = gp.penalized_multinomial_loglik(X, y, b0n, Bn, 0.1)
    assert after["penalized_loglik"] > before["penalized_loglik"]
    # the IRLS weights are p(1-p), so they never exceed 1/4
    assert all(0 < w <= 0.25 + 1e-12 for w in upd["weights"])


def test_eq_7_11_poisson_pmf_and_fit():
    # pmf against the closed form
    assert abs(gp.poisson_pmf(2, 3.0)
               - math.exp(-3.0) * 9.0 / 2.0) < 1e-12
    assert abs(sum(gp.poisson_pmf(k, 2.5) for k in range(60))
               - 1.0) < 1e-10
    # unpenalized fit recovers a known log-linear mean
    rng = gp.np.random.default_rng(8)
    n = 300
    X = [[float(rng.uniform(-1, 1))] for _ in range(n)]
    def rpois(lam):
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= float(rng.uniform(0, 1))
            if p <= L:
                return k
            k += 1
    y = [rpois(math.exp(0.7 + 1.2 * row[0])) for row in X]
    r = mvsml_bayesian_regression_pt2_eq_7_11(X, y, lam=0.0)
    assert abs(r["beta"][0] - 0.7) < 0.25
    assert abs(r["beta"][1] - 1.2) < 0.25
    assert r["iterations"] < 100
    # the ridge penalty shrinks the slope toward zero
    pen = mvsml_bayesian_regression_pt2_eq_7_11(X, y, lam=200.0)
    assert abs(pen["beta"][1]) < abs(r["beta"][1])
    assert pen["penalized_loglik"] < pen["loglik"]


def test_polya_gamma_draw_has_the_right_mean():
    # PG(b, c) has mean (b / (2c)) tanh(c/2); PG(b, 0) has mean b/4
    rng = gp.np.random.default_rng(2)
    draws = [gp._rpolya_gamma(rng, 2.0, 0.0) for _ in range(1500)]
    assert abs(sum(draws) / len(draws) - 0.5) < 0.05
    d2 = [gp._rpolya_gamma(rng, 2.0, 2.0) for _ in range(1500)]
    want = (2.0 / (2.0 * 2.0)) * math.tanh(1.0)
    assert abs(sum(d2) / len(d2) - want) < 0.05


def test_ordinal_logistic_gibbs_recovers_a_signal():
    rng = gp.np.random.default_rng(11)
    n = 100
    X = [[float(rng.normal(0, 1))] for _ in range(n)]
    y = []
    for row in X:
        lat = -1.5 * row[0] + float(rng.normal(0, 1))
        y.append(1 if lat < -0.5 else (2 if lat < 0.7 else 3))
    r = gp.ordinal_logistic_gibbs(y, X, n_iter=400, burn_in=150)
    assert r["n_categories"] == 3
    assert r["gamma"][0] < r["gamma"][1]
    assert r["beta"][0] > 0.3
    assert r["omega_mean"] > 0
