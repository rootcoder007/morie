"""Kosorok Z/M-estimator theory and semiparametric efficiency."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr021 import kosorok_residual_edf
from morie.fn.ksr023 import kosorok_cox_score_process
from morie.fn.ksr046 import kosorok_z_consistency
from morie.fn.ksr047 import kosorok_survival_psi
from morie.fn.ksr048 import kosorok_stochastic_equicontinuity
from morie.fn.ksr049 import kosorok_asymptotic_linearity
from morie.fn.ksr054 import kosorok_lipschitz_envelope
from morie.fn.ksr057 import kosorok_m_normality
from morie.fn.ksr066 import kosorok_no_bias
from morie.fn.ksr067 import kosorok_eff_score_consistency
from morie.fn.ksr070 import kosorok_score_operator
from morie.fn.ksr072 import kosorok_semipar_efficiency
from morie.fn.ksr073 import kosorok_joint_convergence


def test_residual_edf_is_a_distribution_function():
    rng = np.random.default_rng(0)
    n = 300
    Z = rng.standard_normal((n, 2))
    beta = np.array([1.0, -0.5])
    y = Z @ beta + rng.standard_normal(n)
    out = kosorok_residual_edf(y, Z, beta)
    F = out["F_hat"]
    assert np.all(np.diff(F) >= -1e-12)          # non-decreasing
    assert F[0] > 0 and F[-1] == pytest.approx(1.0)
    # the whole point of the example
    assert out["limit_is_brownian_bridge"] is False
    with pytest.raises(ValueError):
        kosorok_residual_edf(y, Z, np.array([1.0]))


def test_cox_score_is_a_process_that_vanishes_at_the_mle():
    from scipy import optimize
    rng = np.random.default_rng(1)
    n = 300
    z = rng.standard_normal(n)
    t = rng.exponential(1.0, n) / np.exp(0.7 * z)
    ev = (rng.random(n) > 0.2).astype(float)

    def final(b):
        return kosorok_cox_score_process(np.array([b]), z, t, ev)["U_final"][0]

    bhat = optimize.brentq(final, -3.0, 3.0)
    assert abs(bhat - 0.7) < 0.3
    out = kosorok_cox_score_process(np.array([bhat]), z, t, ev)
    # the defining property: the score vanishes at the estimate
    assert abs(out["U_final"][0]) < 1e-8
    # and it is a PROCESS, not a number
    assert out["is_process"] is True
    assert out["U"].shape[0] == out["t_grid"].size
    # it is a genuine path: nonzero in the middle, returning to the
    # root at the end -- which is precisely why weak convergence of
    # the PROCESS, not just of its endpoint, is what is needed
    assert np.max(np.abs(out["U"])) > 1e-6


def test_survival_psi_computes_the_printed_functional():
    # Eq. (2.11) is implemented exactly as printed, with S0, L and G
    # SUPPLIED. Plausible empirical stand-ins for them do not make
    # Kaplan-Meier a root, so they are not guessed at here.
    g = np.linspace(0.1, 2.0, 40)
    S0 = np.exp(-g)
    L = np.exp(-0.5 * g)
    G = 1 - np.exp(-0.3 * g)
    out = kosorok_survival_psi(S0, g, S0, L, G)
    # at S = S0 the ratio S0/S is one, so the integral term collapses
    # to G(t) and the map reduces to S0(L + G - 1) exactly
    expected = S0 * (L + G - 1.0)
    assert np.allclose(out["psi"], expected, atol=1e-12)
    assert out["components_supplied"] is True
    assert out["sup_norm"] == pytest.approx(float(np.max(np.abs(expected))))
    with pytest.raises(ValueError):
        kosorok_survival_psi(S0[:5], g, S0, L, G)


def test_z_consistency_needs_both_conditions():
    # Psi(theta) = theta - 1 (root at 1); Psi_n adds a shrinking
    # perturbation, so both conditions improve with n
    def psi(th, t):
        return float(np.asarray(th).ravel()[0]) - 1.0

    # the perturbation must SHRINK with n for the uniform-convergence
    # condition to hold; a fixed one leaves sup||Psi_n - Psi|| constant
    seq = [1.5, 1.2, 1.05, 1.01]
    eps = {1.5: 0.5, 1.2: 0.2, 1.05: 0.05, 1.01: 0.01}

    def psi_n(th, t):
        return psi(th, t) + eps[float(np.asarray(th).ravel()[0])] * np.sin(10 * t)

    out = kosorok_z_consistency(psi_n, psi, seq, 1.0)
    assert out["uniform_convergence"] is True
    assert out["near_root"] is True
    assert out["consistent"] is True
    assert out["both_needed"] is True
    # a fixed perturbation breaks uniform convergence, and then the
    # theorem gives nothing even though the near-root condition holds
    fixed = kosorok_z_consistency(lambda th, t: psi(th, t) + 0.3 * np.sin(10 * t),
                                  psi, seq, 1.0)
    assert fixed["uniform_convergence"] is False
    assert fixed["consistent"] is False
    with pytest.raises(ValueError):
        kosorok_z_consistency(psi_n, psi, [1.0], 1.0)


def test_stochastic_equicontinuity_ratio_shrinks():
    def psi(th, t):
        return float(np.asarray(th).ravel()[0]) * t

    def psi_n(th, t):
        return psi(th, t) + 0.01 * t   # a fixed, theta-free perturbation

    ns = [100, 400, 1600, 6400]
    ths = [1.0 + 1.0 / np.sqrt(m) for m in ns]
    out = kosorok_stochastic_equicontinuity(psi_n, psi, ths, 1.0, ns)
    # a theta-free perturbation cancels exactly in the difference, so
    # the numerator is identically zero and the condition holds in the
    # strongest possible way
    assert np.allclose(out["numerator"], 0.0)
    assert np.allclose(out["ratio"], 0.0)
    assert out["denominator_is_essential"] is True
    # the denominator is 1 + sqrt(n)||theta_n - theta_0||, which is
    # exactly 2 here since theta_n - theta_0 = n^{-1/2}
    assert np.allclose(out["denominator"], 2.0)


def test_asymptotic_linearity_reports_the_residual_and_invertibility():
    out = kosorok_asymptotic_linearity(
        np.array([[2.0]]), lambda th, t: 0.1 * t, lambda th, t: 0.0,
        np.array([1.01]), np.array([1.0]), 400)
    assert out["derivative_invertible"] is True
    assert np.isfinite(out["residual_norm"])
    singular = kosorok_asymptotic_linearity(
        np.array([[0.0]]), lambda th, t: 0.0, lambda th, t: 0.0,
        np.array([1.0]), np.array([1.0]), 100)
    assert singular["derivative_invertible"] is False


def test_lipschitz_envelope_holds_for_smooth_and_fails_for_a_jump():
    x = np.linspace(-2, 2, 101)
    smooth = kosorok_lipschitz_envelope(
        lambda th, v: float(np.sin(th[0] * v)), lambda v: abs(v) + 1e-9,
        [np.array([0.5]), np.array([0.7]), np.array([1.1])], x)
    assert smooth["holds"] is True
    # an indicator in theta has no envelope: this is exactly why
    # maximum-score estimators are n^{-1/3}, not root-n
    # the grid must contain a point BETWEEN the two thresholds, or
    # the indicators never differ and the violation is invisible
    xj = np.array([-1.0, 0.0005, 1.0])
    jump = kosorok_lipschitz_envelope(
        lambda th, v: float(v > th[0]), lambda v: 1e-6,
        [np.array([0.0]), np.array([0.001])], xj)
    assert jump["holds"] is False
    assert jump["worst_ratio"] > 1.0


def test_m_normality_is_a_sandwich_not_an_inverse_hessian():
    rng = np.random.default_rng(3)
    S = rng.standard_normal((500, 2))
    # with V supplied and different from Sigma, the sandwich must
    # differ from V^{-1}
    V = np.array([[2.0, 0.0], [0.0, 0.5]])
    out = kosorok_m_normality(S, V=V)
    assert out["information_equality_assumed"] is False
    assert not np.allclose(out["avar"], np.linalg.pinv(V))
    # omitting V assumes the information equality, and says so
    assumed = kosorok_m_normality(S)
    assert assumed["information_equality_assumed"] is True
    assert assumed["information_equality_holds"] is True
    assert np.allclose(assumed["avar"], np.linalg.pinv(assumed["Sigma"]))


def test_no_bias_tolerance_includes_the_distance_term():
    ns = [100, 400, 1600]
    ths = [1.0 + 1.0 / np.sqrt(m) for m in ns]
    out = kosorok_no_bias(lambda th, m: 0.05 / m, ths, 1.0, ns)
    assert out["holds"] is True
    # the tolerance is n^{-1/2} + ||theta_n - theta||, which here is
    # twice n^{-1/2}, not n^{-1/2}
    assert out["tolerance"][0] == pytest.approx(2 * 100 ** -0.5)


def test_efficient_score_conditions_are_reported_separately():
    rng = np.random.default_rng(4)
    true = rng.standard_normal((200, 2))
    close = true + rng.standard_normal((200, 2)) * 0.01
    out = kosorok_eff_score_consistency(close, true)
    assert out["converges"] is True
    assert out["bounded"] is True
    assert out["both_hold"] is True
    assert out["mean_square_difference"] < 0.01
    # a fit can converge and still be badly behaved: the conditions
    # are separate for a reason
    far = true + rng.standard_normal((200, 2)) * 5.0
    assert kosorok_eff_score_consistency(far, true)["converges"] is False
    with pytest.raises(ValueError):
        kosorok_eff_score_consistency(close[:10], true)


def test_score_operator_is_mean_zero_after_centring():
    x = np.linspace(-2, 2, 201)

    def log_p(eta, v):
        return -0.5 * (v - eta) ** 2

    out = kosorok_score_operator(log_p, lambda t, h: t * h, x, h=1.0)
    assert abs(out["mean_after_centring"]) < 1e-10
    # d/dt of -(v - t h)^2/2 at t=0 is h(v - 0) = v
    assert np.allclose(out["score"], x, atol=1e-4)
    with pytest.raises(ValueError):
        kosorok_score_operator(log_p, lambda t, h: t * h, x, h=1.0, step=0.0)


def test_efficient_information_never_exceeds_the_full_information():
    rng = np.random.default_rng(5)
    n = 400
    nuis = rng.standard_normal((n, 2))
    # scores correlated with the nuisance directions: projecting them
    # out must cost information
    S = 0.8 * nuis[:, :1] + rng.standard_normal((n, 1)) * 0.6
    out = kosorok_semipar_efficiency(S, nuisance_scores=nuis)
    assert out["efficient_information"][0, 0] < out["full_information"][0, 0]
    assert out["information_loss"] > 0
    assert out["adaptive"] is False
    # orthogonal scores lose nothing: the adaptive case
    orth = rng.standard_normal((n, 1))
    ad = kosorok_semipar_efficiency(orth, nuisance_scores=nuis)
    assert ad["information_loss"] < 0.2 * out["information_loss"]
    # no nuisance at all is the parametric case
    par = kosorok_semipar_efficiency(S)
    assert np.allclose(par["efficient_information"], par["full_information"])


def test_joint_convergence_keeps_the_correlation_between_blocks():
    rng = np.random.default_rng(6)
    n = 500
    a = rng.standard_normal(n)
    S = np.column_stack([a, 0.9 * a + 0.4 * rng.standard_normal(n)])
    D = np.array([[1.0, 0.2], [0.0, 1.0]])
    out = kosorok_joint_convergence(D, S)
    assert out["jointly"] is True
    assert out["operator_invertible"] is True
    # the two blocks are genuinely correlated: treating them as
    # independent would understate the variability of any function
    # of both
    assert abs(out["correlation"][0, 1]) > 0.5
    assert out["avar"].shape == (2, 2)
    with pytest.raises(ValueError):
        kosorok_joint_convergence(np.eye(3), S)
