"""Known-answer tests for MVSML chapter 7, eq. (7.1)-(7.2)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm085 import mvsml_bayesian_regression_pt2_eq_7_1
from morie.fn.msm089 import mvsml_bayesian_regression_pt2_eq_7_2


def test_ordinal_probabilities_sum_to_one_and_order_correctly():
    # eq. (7.1): p_ic = F(gamma_c + eta) - F(gamma_{c-1} + eta)
    probs = gp.ordinal_probabilities([0.0], [-1.0, 0.5])
    assert abs(sum(probs[0]) - 1.0) < 1e-12
    assert len(probs[0]) == 3
    # hand values against the normal CDF
    assert abs(probs[0][0] - gp._norm_cdf(-1.0)) < 1e-12
    assert abs(probs[0][1]
               - (gp._norm_cdf(0.5) - gp._norm_cdf(-1.0))) < 1e-12
    assert abs(probs[0][2] - (1.0 - gp._norm_cdf(0.5))) < 1e-12
    # the latent variable is L_i = -x_i'beta + eps (p.210), so a
    # larger linear predictor pushes mass DOWN the category scale
    hi = gp.ordinal_probabilities([2.0], [-1.0, 0.5])
    assert hi[0][0] > probs[0][0]
    assert hi[0][2] < probs[0][2]


def test_logistic_link_matches_the_book_formula():
    probs = gp.ordinal_probabilities([0.3], [-0.5, 1.0],
                                     link="logistic")
    F = lambda z: 1.0 / (1.0 + math.exp(-z))
    assert abs(probs[0][0] - F(-0.5 + 0.3)) < 1e-12
    assert abs(probs[0][1] - (F(1.0 + 0.3) - F(-0.5 + 0.3))) < 1e-12
    assert abs(sum(probs[0]) - 1.0) < 1e-12


def test_norm_ppf_inverts_norm_cdf():
    for u in (0.001, 0.02, 0.25, 0.5, 0.75, 0.98, 0.999):
        assert abs(gp._norm_cdf(gp._norm_ppf(u)) - u) < 1e-9


def test_truncated_normal_stays_in_its_interval():
    rng = gp.np.random.default_rng(4)
    draws = [gp._rtruncnorm(rng, 0.0, 1.0, 0.5, 1.5)
             for _ in range(500)]
    assert all(0.5 <= d <= 1.5 for d in draws)
    # the mean of a N(0,1) truncated to (0.5, 1.5) is about 0.90
    assert abs(sum(draws) / len(draws) - 0.90) < 0.05


def test_eq_7_1_recovers_an_ordinal_signal():
    rng = gp.np.random.default_rng(11)
    n = 120
    X = [[float(rng.normal(0, 1))] for _ in range(n)]
    # latent L = -x*beta + eps with beta = 1.5 cut at -0.5 and 0.7
    y = []
    for row in X:
        lat = -1.5 * row[0] + float(rng.normal(0, 1))
        y.append(1 if lat < -0.5 else (2 if lat < 0.7 else 3))
    r = mvsml_bayesian_regression_pt2_eq_7_1(y, X, n_iter=900,
                                             burn_in=300)
    assert r["n_categories"] == 3
    assert len(r["gamma"]) == 2
    assert r["gamma"][0] < r["gamma"][1]        # ordered thresholds
    assert r["beta"][0] > 0.5                   # sign per l = -x'beta
    assert all(abs(sum(p) - 1.0) < 1e-9 for p in r["probabilities"])


def test_eq_7_2_ordinal_gblup_orders_genotypes():
    rng = gp.np.random.default_rng(5)
    n = 20
    A = [[float(rng.normal(0, 1)) for _ in range(30)]
         for _ in range(n)]
    G = gp.grm_vanraden_method3(A)
    G = [[G[i][j] + (0.5 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    # first half in the low category, second half in the high one
    y = [1] * (n // 2) + [3] * (n // 2)
    r = mvsml_bayesian_regression_pt2_eq_7_2(y, G, n_iter=600,
                                             burn_in=200)
    lo = sum(r["b"][:n // 2]) / (n // 2)
    hi = sum(r["b"][n // 2:]) / (n // 2)
    # l_i is centred at -b_i, so the high category has the lower b
    assert hi < lo
    assert r["sigma2_g"] > 0
    assert r["gamma"][0] < r["gamma"][1]


def test_binary_case_reduces_to_probit_regression():
    # book p.210: "When the response value only takes two values,
    # model (7.1) is reduced to the binary regression model"
    probs = gp.ordinal_probabilities([0.4], [0.0])
    assert len(probs[0]) == 2
    assert abs(probs[0][0] - gp._norm_cdf(0.4)) < 1e-12
    assert abs(probs[0][1] - (1.0 - gp._norm_cdf(0.4))) < 1e-12
