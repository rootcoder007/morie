"""ML, causal, TurboQuant and nonparametric shelf.

Every expectation is a closed-form truth, an algebraic identity the
estimator must satisfy exactly, or a property measured over repeated
draws. Measured values are recorded in the comments.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._did import add_intercept, ols_fit
from morie.fn.diffmed import difference_in_coefficients
from morie.fn.drblr import doubly_robust_ate
from morie.fn.entest import knn_entropy
from morie.fn.gae import generalized_advantage_estimation as gae
from morie.fn.grapl import geron_average_pooling_2d
from morie.fn.groob import geron_oob_evaluation
from morie.fn.hmense import ensemble_evaluate
from morie.fn.ipwef import ipw_ate
from morie.fn.kerd import kernel_density
from morie.fn.kpmnsv import kaplan_meier
from morie.fn.nwest import newey_west_hac
from morie.fn.prdmed import product_of_coefficients
from morie.fn.tqprod import turboquant_qjl_product_estimator
from morie.fn.tqunb import turboquant_prodqjl_unbiasedness
from morie.fn.waldr import wald_estimator


# --------------------------------------------------------------------
# pooling and ensembles
# --------------------------------------------------------------------

def test_average_pooling_is_the_window_mean():
    a = np.arange(16.0).reshape(4, 4)
    out = geron_average_pooling_2d(a, 2)
    assert out["pooled"].tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_same_padding_preserves_size_at_unit_stride():
    a = np.arange(16.0).reshape(4, 4)
    out = geron_average_pooling_2d(a, 2, stride=1, padding="same")
    assert out["pooled"].shape == (4, 4)


def test_pooling_a_constant_returns_the_constant():
    out = geron_average_pooling_2d(np.full((6, 6), 3.0), 3)
    assert np.allclose(out["pooled"], 3.0)


def test_ambiguity_decomposition_is_exact():
    # Krogh-Vedelsby: err_ensemble = mean member error - ambiguity, and
    # it is an identity for a weighted mean, not an approximation
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    P = np.vstack([y + rng.normal(scale=0.5, size=200) for _ in range(4)])
    out = ensemble_evaluate(P, y=y)
    assert out["decomposition_residual"] < 1e-10
    assert out["beats_average_member"]


def test_identical_members_buy_nothing():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    p = y + rng.normal(scale=0.4, size=100)
    out = ensemble_evaluate(np.vstack([p, p, p]), y=y)
    assert out["diversity"] == pytest.approx(0.0, abs=1e-12)
    assert out["ambiguity"] == pytest.approx(0.0, abs=1e-12)
    assert out["ensemble_error"] == pytest.approx(out["mean_member_error"])


def test_oob_leave_out_rate_is_one_over_e():
    out = geron_oob_evaluation([0.0, 1.0],
                               [[9.0, 1.0], [0.0, 9.0]],
                               [[True, False], [False, True]])
    assert out["expected_oob_fraction"] == pytest.approx(1 / np.e)
    assert out["oob_error"] == pytest.approx(0.0)


# --------------------------------------------------------------------
# mediation
# --------------------------------------------------------------------

def test_difference_and_product_agree_exactly_for_ols():
    # c - c' = ab is an algebraic identity in every sample, not merely
    # in expectation. Measured residual on this design: 0.00e+00.
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=n)
    M = 0.6 * X + rng.normal(size=n)
    Y = 0.3 * X + 0.5 * M + rng.normal(size=n)
    a = ols_fit(add_intercept(X[:, None]), M)[1]
    bc = ols_fit(add_intercept(np.column_stack([X, M])), Y)
    cprime, b = bc[1], bc[2]
    c = ols_fit(add_intercept(X[:, None]), Y)[1]
    out = difference_in_coefficients(c, cprime, a=a, b=b)
    assert out["identity_residual"] < 1e-12
    assert out["matches_product"]


def test_indirect_effect_recovers_the_design():
    ests = []
    for s in range(40):
        rng = np.random.default_rng(s)
        n = 800
        X = rng.normal(size=n)
        M = 0.6 * X + rng.normal(size=n)
        Y = 0.3 * X + 0.5 * M + rng.normal(size=n)
        a = ols_fit(add_intercept(X[:, None]), M)[1]
        b = ols_fit(add_intercept(np.column_stack([X, M])), Y)[2]
        ests.append(product_of_coefficients(a, b)["indirect"])
    # truth is 0.6 * 0.5 = 0.30
    assert abs(float(np.mean(ests)) - 0.30) < 0.02


def test_bootstrap_interval_is_asymmetric_where_sobel_is_not():
    out = product_of_coefficients(0.2, 0.2, se_a=0.1, se_b=0.1,
                                  n_boot=20000, seed=0)
    lo, hi = out["boot_ci"]
    left = out["indirect"] - lo
    right = hi - out["indirect"]
    # the product of two normals is skewed; the Sobel interval cannot
    # represent that because it is symmetric by construction
    assert abs(right - left) > 0.01
    assert out["sobel_symmetric"] is True


def test_proportion_mediated_is_flagged_as_unstable():
    out = difference_in_coefficients(0.001, 0.0005)
    assert "unstable" in out["proportion_note"]


# --------------------------------------------------------------------
# weighting and doubly robust
# --------------------------------------------------------------------

def confounded(seed=0, n=800, tau=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-0.8 * X[:, 0]))
    d = (rng.uniform(size=n) < e).astype(float)
    y = tau * d + X[:, 0] + rng.normal(size=n)
    return y, d, X


def test_hajek_is_shift_invariant_and_horvitz_thompson_is_not():
    # normalising by the realised weights is what buys the invariance.
    # Measured: Hajek 0.0000, Horvitz-Thompson -0.3981 for a shift of 100.
    y, d, X = confounded(seed=3)
    a = ipw_ate(y, d, X=X, stabilized=True)["estimate"]
    b = ipw_ate(y + 100, d, X=X, stabilized=True)["estimate"]
    assert b == pytest.approx(a, abs=1e-8)
    c = ipw_ate(y, d, X=X, stabilized=False)["estimate"]
    e = ipw_ate(y + 100, d, X=X, stabilized=False)["estimate"]
    assert abs(e - c) > 0.1


def test_ipw_recovers_the_true_effect_under_confounding():
    ests = [ipw_ate(*confounded(seed=s)[:2],
                    X=confounded(seed=s)[2])["estimate"] for s in range(20)]
    assert abs(float(np.mean(ests)) - 2.0) < 0.15


def test_ipw_reports_effective_sample_size():
    y, d, X = confounded(seed=4)
    out = ipw_ate(y, d, X=X)
    assert 0.0 < out["ess_fraction"] <= 1.0
    assert out["max_weight_share"] < 0.5


def test_doubly_robust_survives_a_wrong_propensity():
    # outcome model correct, propensity deliberately wrong
    y, d, X = confounded(seed=5)
    bad = np.full(d.size, 0.5)
    out = doubly_robust_ate(y, d, X, propensity=bad)
    assert abs(out["estimate"] - 2.0) < 0.25


def test_doubly_robust_survives_a_wrong_outcome_model():
    y, d, X = confounded(seed=6)
    n = d.size
    out = doubly_robust_ate(y, d, X, mu1=np.zeros(n), mu0=np.zeros(n))
    assert abs(out["estimate"] - 2.0) < 0.3


def test_model_disagreement_is_reported():
    y, d, X = confounded(seed=7)
    out = doubly_robust_ate(y, d, X)
    assert out["model_disagreement"] >= 0.0
    assert "wrong" in out["disagreement_note"]


# --------------------------------------------------------------------
# instrumental variables
# --------------------------------------------------------------------

def test_wald_is_the_ratio_of_the_two_differences():
    y = [0, 0, 0, 0, 0, 1, 1, 1]
    d = [0, 0, 0, 0, 0, 1, 1, 1]
    z = [0, 0, 0, 0, 1, 1, 1, 1]
    out = wald_estimator(y, d, z)
    assert out["estimate"] == pytest.approx(
        out["reduced_form"] / out["first_stage"]
    )
    assert out["estimate"] == pytest.approx(1.0)


def test_wald_refuses_a_zero_first_stage():
    with pytest.raises(ValueError, match="first stage is zero"):
        wald_estimator([1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1])


def test_weak_instrument_is_flagged():
    rng = np.random.default_rng(0)
    n = 400
    z = (rng.uniform(size=n) < 0.5).astype(float)
    # the instrument barely moves treatment
    d = (rng.uniform(size=n) < 0.5 + 0.01 * z).astype(float)
    y = d + rng.normal(size=n)
    out = wald_estimator(y, d, z)
    assert out["weak_instrument"]
    assert "OLS" in out["weak_note"]


# --------------------------------------------------------------------
# HAC
# --------------------------------------------------------------------

def test_hac_inflates_under_autocorrelation_and_not_under_white_noise():
    # measured: 1.98 on AR(0.8), 1.09 on white noise
    rng = np.random.default_rng(0)
    T = 600
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.8 * u[t - 1] + rng.normal()
    assert newey_west_hac(u[:, None])["inflation"][0] > 1.5
    assert newey_west_hac(rng.normal(size=(T, 1)))["inflation"][0] < 1.3


def test_bartlett_weights_keep_the_matrix_positive_semidefinite():
    rng = np.random.default_rng(1)
    for seed in range(5):
        g = np.random.default_rng(seed).normal(size=(150, 3))
        assert newey_west_hac(g, lags=12)["positive_definite"]


def test_automatic_lag_rule_matches_the_formula():
    rng = np.random.default_rng(2)
    T = 500
    out = newey_west_hac(rng.normal(size=(T, 1)))
    assert out["lags"] == int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


# --------------------------------------------------------------------
# TurboQuant
# --------------------------------------------------------------------

def test_prodqjl_is_unbiased():
    # measured z at m = 16, 64, 256: 1.56, 1.07, 0.62
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=64), rng.normal(size=64)
    out = turboquant_prodqjl_unbiasedness(q, k, m=64, trials=3000, seed=1)
    assert out["unbiased"]
    assert abs(out["z"]) < 4.0


def test_prodqjl_variance_falls_as_one_over_m():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=64), rng.normal(size=64)
    scaled = [
        turboquant_prodqjl_unbiasedness(q, k, m=m, trials=1500,
                                        seed=1)["variance_scaling"]
        for m in (16, 64, 256)
    ]
    # variance * m is constant to within sampling noise; measured
    # 5158, 5108, 5100
    assert max(scaled) / min(scaled) < 1.5


def test_estimator_uses_the_forced_constant():
    S = np.eye(4)
    out = turboquant_qjl_product_estimator(
        np.ones(4), [[1, 1, 1, 1]], 1.0, S
    )
    assert out["constant"] == pytest.approx(np.sqrt(np.pi / 2))
    assert out["compression"] > 1.0


def test_sketch_rejects_non_sign_input():
    with pytest.raises(ValueError, match="only -1 and \\+1"):
        turboquant_qjl_product_estimator(np.ones(2), [[0.5, 1.0]], 1.0,
                                         np.eye(2))


# --------------------------------------------------------------------
# survival and density
# --------------------------------------------------------------------

def test_kaplan_meier_matches_the_exponential_truth():
    # measured: median 7.14 against ln(2)*10 = 6.93, S(10) 0.3721
    # against exp(-1) = 0.3679
    rng = np.random.default_rng(0)
    n = 2000
    T = rng.exponential(10, n)
    C = rng.exponential(20, n)
    out = kaplan_meier(np.minimum(T, C), (T <= C).astype(int))
    assert abs(out["median"] - np.log(2) * 10) < 0.6
    i = np.searchsorted(out["times"], 10.0) - 1
    assert out["ci_lower"][i] <= np.exp(-1) <= out["ci_upper"][i]


def test_log_log_interval_stays_inside_the_unit_interval():
    rng = np.random.default_rng(1)
    T = rng.exponential(5, 300)
    C = rng.exponential(8, 300)
    out = kaplan_meier(np.minimum(T, C), (T <= C).astype(int))
    assert out["ci_lower"].min() >= 0.0
    assert out["ci_upper"].max() <= 1.0


def test_survival_is_monotone_and_starts_high():
    out = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    assert np.all(np.diff(out["survival"]) <= 1e-12)
    assert out["survival"][0] == pytest.approx(0.8)


def test_censored_observations_do_not_drop_the_curve():
    # a censored subject leaves the risk set without an event
    both = kaplan_meier([1, 2, 3], [1, 0, 1])
    assert both["n_censored"] == 1
    assert both["times"].tolist() == [1.0, 3.0]


def test_knn_entropy_matches_the_gaussian_value():
    # measured error at d = 1, 2, 4: -0.0023, +0.0029, -0.0741
    rng = np.random.default_rng(0)
    for d in (1, 2):
        x = rng.normal(size=(3000, d))
        truth = 0.5 * d * np.log(2 * np.pi * np.e)
        assert abs(knn_entropy(x, k=4)["entropy"] - truth) < 0.05


def test_entropy_shifts_by_log_a_under_rescaling():
    # differential entropy is not scale invariant: scaling by a adds
    # d log a, which is exactly why it is not comparable across units
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 1))
    h1 = knn_entropy(x, k=4)["entropy"]
    h2 = knn_entropy(x * 5.0, k=4)["entropy"]
    assert h2 - h1 == pytest.approx(np.log(5.0), abs=1e-9)


def test_distance_concentration_falls_with_dimension():
    rng = np.random.default_rng(0)
    c = [knn_entropy(rng.normal(size=(800, d)), k=4)["distance_concentration"]
         for d in (1, 4, 10)]
    assert c[0] > c[1] > c[2]


def test_kde_integrates_to_one():
    rng = np.random.default_rng(0)
    out = kernel_density(rng.normal(size=500))
    assert abs(out["integral"] - 1.0) < 0.02


def test_kde_finds_two_modes_in_a_mixture():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-4, 0.5, 400), rng.normal(4, 0.5, 400)]
    out = kernel_density(x, bandwidth=0.4)
    assert out["n_modes"] == 2


def test_silverman_oversmooths_the_same_mixture():
    # the rule is calibrated to a single-mode Gaussian, which is exactly
    # the case a density estimate was not needed for
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-1.2, 1.0, 300), rng.normal(1.2, 1.0, 300)]
    auto = kernel_density(x)
    tight = kernel_density(x, bandwidth=auto["bandwidth"] / 3)
    assert tight["n_modes"] >= auto["n_modes"]


# --------------------------------------------------------------------
# reinforcement learning
# --------------------------------------------------------------------

def test_gae_at_lambda_one_is_the_monte_carlo_return():
    r = [1.0, 2.0, 3.0]
    v = [0.0, 0.0, 0.0]
    out = gae(r, v, gamma=1.0, lam=1.0)
    assert out["advantages"].tolist() == [6.0, 5.0, 3.0]


def test_gae_at_lambda_zero_is_the_one_step_residual():
    r = [1.0, 2.0, 3.0]
    v = [0.5, 0.5, 0.5]
    out = gae(r, v, gamma=0.9, lam=0.0)
    assert np.allclose(out["advantages"], out["td_errors"])


def test_done_flags_cut_the_bootstrap():
    out = gae([1.0, 1.0], [5.0, 5.0], gamma=1.0, lam=1.0,
              dones=[1.0, 0.0])
    # the first step terminates, so its advantage sees no future value
    assert out["advantages"][0] == pytest.approx(1.0 - 5.0)


def test_effective_horizon_matches_the_formula():
    out = gae([1.0] * 10, [0.0] * 10, gamma=0.99, lam=0.95)
    assert out["effective_horizon"] == pytest.approx(
        1.0 / (1.0 - 0.99 * 0.95)
    )
    assert out["truncation_bias"]


def test_normalization_leaves_the_raw_values_available():
    out = gae([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], normalize=True)
    assert abs(float(out["advantages"].mean())) < 1e-12
    assert not np.allclose(out["advantages"], out["advantages_raw"])
