"""Kosorok shelf: empirical process core + Ch 1-3 modules.

Anchors PDF-verified in Kosorok (2008): bridge covariance
F(s^t)-F(s)F(t), LIL eq. (2.21) bound 1/2, Chung liminf pi/2."""

import numpy as np
import pytest
from scipy import stats

from morie.fn._kosorok import (
    bootstrap_multiplier_process, bracketing_number_monotone, bridge_cov,
    cox_score, covering_number_grid, empirical_df, empirical_process,
    hadamard_derivative, sup_norm,
)
from morie.fn.ksr020 import kosorok_ch1_linear_regression_model
from morie.fn.ksr024 import kosorok_ch1_partly_linear_logistic
from morie.fn.ksr026 import kosorok_ch2_empirical_distribution_function
from morie.fn.ksr027 import kosorok_ch2_law_large_numbers_pointwise
from morie.fn.ksr028 import kosorok_ch2_glivenko_cantelli_classical
from morie.fn.ksr030 import kosorok_ch2_brownian_bridge_covariance
from morie.fn.ksr058 import kosorok_ch2_law_iterated_logarithm
from morie.fn.ksr060 import kosorok_ch2_u_process_measure
from morie.fn.ksr063 import kosorok_ch3_cox_efficient_score_beta
from morie.fn.ksr068 import kosorok_ch3_cox_profile_score
from morie.fn.ksr071 import kosorok_ch3_log_profile_expansion


def test_edf_and_empirical_process_are_the_textbook_objects():
    X = np.array([0.1, 0.4, 0.4, 0.7, 0.9])
    assert empirical_df(X, [0.4])[0] == pytest.approx(0.6)  # <= is inclusive
    assert empirical_df(X, [0.0])[0] == 0.0
    assert empirical_df(X, [1.0])[0] == 1.0
    out = kosorok_ch2_empirical_distribution_function(X)
    assert out["F_n"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(out["F_n"]) >= 0)  # monotone
    # G_n = sqrt(n)(F_n - F) has mean 0 under the true F
    rng = np.random.default_rng(0)
    U = rng.random(2000)
    g = empirical_process(U, [0.3, 0.6])
    assert abs(g).max() < 4.0  # within a few standard deviations
    with pytest.raises(ValueError):
        kosorok_ch2_empirical_distribution_function([], None)


def test_bridge_covariance_matches_simulation_and_is_tied_down():
    # PDF-verified: cov[G(s), G(t)] = F(s ^ t) - F(s)F(t)
    assert bridge_cov(0.3, 0.7) == pytest.approx(0.3 - 0.21)
    assert bridge_cov(0.5, 0.5) == pytest.approx(0.25)
    assert bridge_cov(0.0, 0.5) == pytest.approx(0.0)  # tied down at 0
    assert bridge_cov(1.0, 0.5) == pytest.approx(0.0)  # and at 1
    # Monte Carlo: the empirical process covariance converges to it
    rng = np.random.default_rng(1)
    s, t = 0.3, 0.7
    vals = np.array([empirical_process(rng.random(400), [s, t]) for _ in range(1500)])
    emp_cov = float(np.cov(vals[:, 0], vals[:, 1])[0, 1])
    assert emp_cov == pytest.approx(bridge_cov(s, t), abs=0.05)
    out = kosorok_ch2_brownian_bridge_covariance(s, t)
    assert out["covariance"] == pytest.approx(0.09)
    assert out["variance_s"] == pytest.approx(0.21)


def test_glivenko_cantelli_and_pointwise_law_converge():
    rng = np.random.default_rng(2)
    X = rng.random(4000)
    gc = kosorok_ch2_glivenko_cantelli_classical(X)
    # sup distance shrinks with n, and obeys the DKW bound
    assert gc["sup_distance"][-1] < gc["sup_distance"][0]
    assert np.all(gc["dkw_bound"] <= 1.0)
    # the sup is taken at the jumps: it must be at least the value any
    # fixed grid would report
    grid = np.linspace(0, 1, 50)
    grid_sup = np.abs(empirical_process(X, grid)).max()
    assert sup_norm(X) >= grid_sup - 1e-9
    pw = kosorok_ch2_law_large_numbers_pointwise(X, 0.4)
    assert pw["shrinking"] is True
    assert pw["F_t"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        kosorok_ch2_law_large_numbers_pointwise(X[:3], 0.4)


def test_lil_reports_the_finite_n_gap_honestly():
    rng = np.random.default_rng(3)
    out = kosorok_ch2_law_iterated_logarithm(rng.random(5000))
    assert out["lil_bound"] == 0.5
    assert out["chung_liminf_constant"] == pytest.approx(np.pi / 2)
    # at any realistic n the ratio sits well under 1/2: log log 5000 ~ 2.1
    assert 0 < out["lil_ratio"] < 0.5
    assert out["loglog_term"] == pytest.approx(np.sqrt(2 * np.log(np.log(5000))))
    with pytest.raises(ValueError):
        kosorok_ch2_law_iterated_logarithm(rng.random(4))


def test_multiplier_bootstrap_is_centred_and_matches_the_bridge():
    rng = np.random.default_rng(4)
    X = rng.random(300)
    t = np.array([0.25, 0.5, 0.75])
    draws = np.array([
        bootstrap_multiplier_process(X, t, rng=rng) for _ in range(1200)
    ])
    # mean-centred weights => the process is centred
    assert np.abs(draws.mean(axis=0)).max() < 0.15
    # and reproduces the bridge covariance structure
    emp = np.cov(draws.T)
    assert emp[0, 0] == pytest.approx(bridge_cov(0.25, 0.25), abs=0.06)
    assert emp[0, 2] == pytest.approx(bridge_cov(0.25, 0.75), abs=0.06)
    with pytest.raises(ValueError):
        bootstrap_multiplier_process(X, t, weights=np.ones(5))


def test_entropy_helpers_are_monotone_and_finite():
    # monotone indicators: N_[](eps) <= ceil(1/eps) + 1, finite for all eps
    assert bracketing_number_monotone(0.1) == 11
    assert bracketing_number_monotone(0.5) == 3
    assert bracketing_number_monotone(0.01) > bracketing_number_monotone(0.1)
    with pytest.raises(ValueError):
        bracketing_number_monotone(0.0)
    # covering numbers decrease as the radius grows
    rng = np.random.default_rng(5)
    P = rng.random((200, 2))
    n_small = covering_number_grid(P, 0.05)
    n_big = covering_number_grid(P, 0.4)
    assert n_small > n_big >= 1
    assert covering_number_grid(P, 10.0) == 1


def test_hadamard_derivative_recovers_a_known_derivative():
    # phi(theta) = theta^2 has derivative 2 theta h
    der, drift, ok = hadamard_derivative(lambda th: th**2, 3.0, 1.0)
    assert float(der) == pytest.approx(6.0, abs=1e-3)
    assert ok is True
    # directional, not central: |.| at 0 in direction +1 has one-sided
    # derivative 1, which a central difference would report as 0
    der_abs, _, ok_abs = hadamard_derivative(lambda th: abs(th), 0.0, 1.0)
    assert float(der_abs) == pytest.approx(1.0, abs=1e-9)
    assert ok_abs is True
    der_neg, _, _ = hadamard_derivative(lambda th: abs(th), 0.0, -1.0)
    assert float(der_neg) == pytest.approx(1.0, abs=1e-9)


def _cox_data(seed=0, n=300, beta=0.8):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 1))
    haz = np.exp(Z[:, 0] * beta)
    T = rng.exponential(1.0 / haz)
    C = rng.exponential(2.0, size=n)
    return Z, np.minimum(T, C), (T <= C).astype(float)


def test_cox_score_vanishes_at_the_partial_likelihood_estimator():
    Z, time, event = _cox_data()
    prof = kosorok_ch3_cox_profile_score(Z=Z, time=time, event=event)
    assert np.abs(prof["score_at_root"]).max() < 1e-6  # the root really is one
    assert prof["beta_hat"][0] == pytest.approx(0.8, abs=0.25)  # recovers beta
    assert prof["information"][0, 0] > 0
    # the efficient score module agrees at the same beta
    eff = kosorok_ch3_cox_efficient_score_beta(
        Z, time=time, event=event, beta=prof["beta_hat"]
    )
    assert np.abs(eff["score"]).max() < 1e-6
    assert eff["efficient_information"] == pytest.approx(prof["information"])
    with pytest.raises(ValueError):
        kosorok_ch3_cox_efficient_score_beta(Z, time=time, event=None)


def test_cox_information_matches_the_sampling_variance():
    # the efficient information is the inverse asymptotic variance:
    # sd(beta_hat) over replications should match 1/sqrt(I)
    hats, infos = [], []
    for s in range(25):
        Z, time, event = _cox_data(seed=s, n=400)
        out = kosorok_ch3_cox_profile_score(Z=Z, time=time, event=event)
        hats.append(out["beta_hat"][0])
        infos.append(out["information"][0, 0])
    emp_sd = float(np.std(hats, ddof=1))
    model_sd = float(np.mean([1 / np.sqrt(i) for i in infos]))
    assert emp_sd == pytest.approx(model_sd, rel=0.4)  # measured close


def test_profile_expansion_is_quadratic_and_calibrated():
    out = kosorok_ch3_log_profile_expansion(
        theta_bar_n=[0.5], theta_hat_n=[0.4], I_tilde=[[4.0]], n=100
    )
    assert out["quadratic_term"] == pytest.approx(0.5 * 100 * 0.01 * 4.0)
    assert out["lrt_statistic"] == pytest.approx(2 * out["quadratic_term"])
    # at the maximiser the drop is exactly zero
    zero = kosorok_ch3_log_profile_expansion([0.4], [0.4], [[4.0]], n=100)
    assert zero["quadratic_term"] == 0.0
    with pytest.raises(ValueError):
        kosorok_ch3_log_profile_expansion([0.5], [0.4], [[4.0]])  # n missing


def test_linear_model_assumption_checks_fire_correctly():
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((300, 2))
    Y = Z @ np.array([1.5, -0.5]) + rng.standard_normal(300) * 0.4
    out = kosorok_ch1_linear_regression_model(Y, Z)
    assert out["beta"] == pytest.approx([1.5, -0.5], abs=0.1)
    assert out["bounded_cond_var"] is True  # homoscedastic by construction
    # heteroscedastic data: the conditional-variance check must notice
    Yh = Z @ np.array([1.5, -0.5]) + rng.standard_normal(300) * np.exp(2 * Z[:, 0])
    assert kosorok_ch1_linear_regression_model(Yh, Z)["cond_var_ratio"] > \
        out["cond_var_ratio"]
    with pytest.raises(ValueError):
        kosorok_ch1_linear_regression_model(Y[:10], Z)  # length mismatch


def test_partly_linear_logistic_recovers_beta_despite_the_nuisance():
    rng = np.random.default_rng(7)
    n = 800
    Z = rng.standard_normal((n, 1))
    U = rng.random(n)
    eta = np.sin(3 * U) * 2.0  # genuinely nonlinear nuisance
    p = 1 / (1 + np.exp(-(1.2 * Z[:, 0] + eta)))
    Y = rng.binomial(1, p)
    out = kosorok_ch1_partly_linear_logistic(Y, Z, U, df=6)
    assert out["beta"][0] == pytest.approx(1.2, abs=0.35)  # root-n part recovered
    assert out["eta_fitted"].size == n
    assert np.isfinite(out["loglik"])
    with pytest.raises(ValueError):
        kosorok_ch1_partly_linear_logistic(np.full(50, 2.0), Z[:50], U[:50])


def test_u_process_matches_a_known_u_statistic():
    # the order-2 kernel |x - y| / 2 has U-statistic = Gini mean difference / 2
    rng = np.random.default_rng(8)
    X = rng.random(60)
    out = kosorok_ch2_u_process_measure(lambda a, b: abs(a - b) / 2, X, m=2)
    from itertools import combinations

    brute = np.mean([abs(a - b) / 2 for a, b in combinations(X, 2)])
    assert out["U"] == pytest.approx(brute)
    assert out["n_subsets"] == 60 * 59 // 2
    assert out["zeta1"] > 0  # summands are dependent
    assert out["hajek_var"] == pytest.approx(4 * out["zeta1"] / 60)
    with pytest.raises(ValueError):
        kosorok_ch2_u_process_measure(lambda a, b: a + b, rng.random(400), m=3)
