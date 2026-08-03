"""Tests for the marginal structural models (Robins, Hernan &
Brumback 2000; Hernan & Robins 2020).

The decisive check is the one an MSM exists for: when a time-varying
confounder affects both treatment and outcome, the unweighted
regression is biased and the IPT-weighted one recovers the causal
parameter.
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msmlin import msm_linear
from morie.fn.msmlog import msm_logistic
from morie.fn.msmpoi import msm_poisson
from morie.fn.msmnbi import msm_negative_binomial
from morie.fn.msmcox import msm_cox_marginal
from morie.fn.msmaft import msm_accelerated_failure
from morie.fn.msmgmm import msm_gmm_estimator
from morie.fn.msmtve import msm_time_varying_exposure


def _confounded(n=4000, seed=5, beta_a=2.0):
    """Two periods.  L1 is a time-varying confounder of the second
    treatment: it affects both A2 and Y, so the unweighted regression
    of Y on the treatment history is confounded.

    L1 is drawn independently of A1 on purpose.  If A1 caused L1, then
    L1 would be a *mediator* of the first-period treatment and the two
    periods would have different total effects (2 and 2 + 3 x 0.6),
    which a one-coefficient MSM cannot represent -- the model, not the
    estimator, would be at fault.  Here both periods have effect
    beta_a, so beta_a is the estimand.
    """
    rng = gp.np.random.default_rng(seed)
    A, Y, W = [], [], []
    for _ in range(n):
        a1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        l1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        p2 = 0.8 if l1 else 0.2
        a2 = 1.0 if float(rng.uniform(0, 1)) < p2 else 0.0
        y = beta_a * (a1 + a2) + 3.0 * l1 + float(rng.normal(0, 1))
        # sw = prod_k P(A_k | A-bar_{k-1}) / P(A_k | A-bar_{k-1}, L-bar_k)
        # numerator conditions on treatment history only (Robins,
        # Hernan & Brumback 2000, eq. 17); marginally P(A2=1) = 0.5
        num = 0.5 * (0.5 if a2 else 0.5)
        den = 0.5 * (p2 if a2 else 1.0 - p2)
        A.append([a1, a2])
        Y.append(y)
        W.append(num / den)
    return A, Y, W


def test_iptw_removes_confounding_where_plain_regression_fails():
    A, Y, W = _confounded()
    naive = msm_linear(Y, A)["beta_a"]
    weighted = msm_linear(Y, A, weights=W)["beta_a"]
    # the unweighted fit is biased upward by the L1 pathway
    assert naive > 2.5
    # the weighted fit recovers the causal parameter
    assert abs(weighted - 2.0) < 0.25
    assert abs(weighted - 2.0) < abs(naive - 2.0)


def test_linear_msm_reduces_to_ols_with_unit_weights():
    A, Y, _ = _confounded(n=200, seed=8)
    r = msm_linear(Y, A)
    d = gp.msm_design(A)
    ols = gp.ols_fit(d["X"], Y, add_intercept=False)
    for a, b in zip(r["beta"], ols["beta"]):
        assert abs(a - b) < 1e-8


def test_logistic_msm_recovers_a_known_odds_ratio():
    rng = gp.np.random.default_rng(3)
    A, Y = [], []
    for _ in range(6000):
        a1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        a2 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        eta = -1.0 + 0.8 * (a1 + a2)
        p = 1.0 / (1.0 + math.exp(-eta))
        A.append([a1, a2])
        Y.append(1.0 if float(rng.uniform(0, 1)) < p else 0.0)
    r = msm_logistic(Y, A)
    assert abs(r["beta"][1] - 0.8) < 0.15
    assert abs(r["odds_ratio"] - math.exp(0.8)) < 0.3


def test_poisson_msm_recovers_a_known_rate_ratio():
    rng = gp.np.random.default_rng(4)

    def rpois(lam):
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= float(rng.uniform(0, 1))
            if p <= L:
                return float(k)
            k += 1

    A, Y = [], []
    for _ in range(4000):
        a1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        a2 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        Y.append(rpois(math.exp(0.5 + 0.4 * (a1 + a2))))
        A.append([a1, a2])
    r = msm_poisson(Y, A)
    assert abs(r["beta"][1] - 0.4) < 0.08
    assert abs(r["rate_ratio"] - math.exp(0.4)) < 0.12


def test_negative_binomial_shares_the_mean_but_inflates_variance():
    A, Y, _ = _confounded(n=400, seed=6)
    Y = [abs(round(v)) for v in Y]
    pois = msm_poisson(Y, A)
    nb = msm_negative_binomial(Y, A, alpha=0.5)
    assert abs(pois["beta"][1] - nb["beta"][1]) < 1e-9
    # V(mu) = mu + alpha mu^2 exceeds the Poisson variance mu
    for m, v in zip(pois["fitted"], nb["variance"]):
        assert v > m - 1e-12


def test_cox_msm_recovers_a_known_hazard_ratio():
    rng = gp.np.random.default_rng(7)
    T, E, A = [], [], []
    for _ in range(3000):
        a1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        a2 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        lam = math.exp(0.6 * (a1 + a2))
        t = -math.log(max(float(rng.uniform(0, 1)), 1e-12)) / lam
        T.append(t)
        E.append(1.0)
        A.append([a1, a2])
    r = msm_cox_marginal(T, E, A)
    assert abs(r["beta"] - 0.6) < 0.12
    assert abs(r["hazard_ratio"] - math.exp(0.6)) < 0.25


def test_aft_recovers_a_known_time_ratio():
    rng = gp.np.random.default_rng(9)
    T, E, A = [], [], []
    for _ in range(2000):
        a1 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        a2 = 1.0 if float(rng.uniform(0, 1)) < 0.5 else 0.0
        logt = 1.0 - 0.5 * (a1 + a2) + 0.3 * float(rng.normal(0, 1))
        T.append(math.exp(logt))
        E.append(1.0)
        A.append([a1, a2])
    r = msm_accelerated_failure(T, E, A)
    assert abs(r["beta"][1] + 0.5) < 0.05
    assert r["n_uncensored"] == 2000


def test_aft_ignores_censored_observations():
    T = [1.0, 2.0, 3.0, 4.0]
    E = [1.0, 0.0, 1.0, 0.0]
    A = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    r = msm_accelerated_failure(T, E, A)
    assert r["n_uncensored"] == 2


def test_gmm_is_exactly_identified_and_matches_weighted_ls():
    A, Y, W = _confounded(n=500, seed=11)
    g = msm_gmm_estimator(Y, A, weights=W)
    lin = msm_linear(Y, A, weights=W)
    for a, b in zip(g["beta"], lin["beta"]):
        assert abs(a - b) < 1e-7
    # the moment conditions are satisfied at the solution
    assert max(abs(v) for v in g["moments"]) < 1e-7


def test_time_varying_exposure_reports_weight_diagnostics():
    A, Y, W = _confounded(n=800, seed=13)
    r = msm_time_varying_exposure(Y, A, weights=W)
    assert abs(r["beta"][1] - 2.0) < 0.4
    # stabilized weights have mean near one (Cole & Hernan 2008)
    assert 0.7 < r["weight_mean"] < 1.4
    assert r["weight_max"] >= r["weight_mean"]


def test_weighted_glm_families_agree_with_unweighted_fits():
    rng = gp.np.random.default_rng(2)
    X = [[1.0, float(rng.normal(0, 1))] for _ in range(300)]
    y = [1.0 if float(rng.uniform(0, 1))
         < 1.0 / (1.0 + math.exp(-(0.3 + 0.7 * row[1]))) else 0.0
         for row in X]
    a = gp.msm_weighted_glm(y, X, family="binomial")
    b = gp.msm_weighted_glm(y, X, weights=[1.0] * 300,
                            family="binomial")
    for u, v in zip(a["beta"], b["beta"]):
        assert abs(u - v) < 1e-12
    # doubling every weight cannot move the estimate
    c = gp.msm_weighted_glm(y, X, weights=[2.0] * 300,
                            family="binomial")
    for u, v in zip(a["beta"], c["beta"]):
        assert abs(u - v) < 1e-8
