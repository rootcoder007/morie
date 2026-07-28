"""Kosorok tranche 4: bootstrap Donsker, quantile Hadamard bounds,
DQM/LAN and efficient influence functions."""

import numpy as np
import pytest
from scipy import stats

from morie.fn.ksr040 import kosorok_ch2_bootstrap_donsker_iff
from morie.fn.ksr041 import kosorok_ch2_bootstrap_donsker_almost_sure
from morie.fn.ksr043 import kosorok_ch2_quantile_hadamard_inequality
from morie.fn.ksr044 import kosorok_ch2_quantile_taylor_bounds
from morie.fn.ksr045 import kosorok_ch2_functional_delta_bootstrap
from morie.fn.ksr061 import kosorok_ch3_differentiable_quadratic_mean
from morie.fn.ksr062 import kosorok_ch3_pathwise_derivative
from morie.fn.ksr065 import kosorok_ch3_efficient_influence_general


def test_bootstrap_reproduces_the_bridge_covariance():
    rng = np.random.default_rng(0)
    X = rng.random(400)
    out = kosorok_ch2_bootstrap_donsker_iff(X, n_boot=600, rng=rng)
    # indicators are Donsker, so the bootstrap covariance must match
    # the bridge: t(1-t) on the diagonal, s(1-t) off it
    assert out["bootstrap_cov"][1, 1] == pytest.approx(0.25, abs=0.06)
    assert out["max_abs_gap"] < 0.08
    assert out["bridge_cov"][0, 2] == pytest.approx(0.25 * 0.25)
    with pytest.raises(ValueError):
        kosorok_ch2_bootstrap_donsker_iff(X[:5])
    with pytest.raises(ValueError):
        kosorok_ch2_bootstrap_donsker_iff(X, n_boot=3)


def test_almost_sure_version_carries_the_extra_envelope_condition():
    rng = np.random.default_rng(1)
    X = rng.random(300)
    out = kosorok_ch2_bootstrap_donsker_almost_sure(X, n_boot=300, rng=rng)
    assert out["envelope_condition_met"] is True
    assert out["both_conditions_met"] is True
    # an infinite envelope moment breaks the a.s. version even though
    # the in-probability version (ksr040) is unaffected
    bad = kosorok_ch2_bootstrap_donsker_almost_sure(
        X, n_boot=300, rng=rng, envelope_sq_mean=np.inf
    )
    assert bad["envelope_condition_met"] is False
    assert bad["both_conditions_met"] is False


def test_quantile_sandwich_brackets_the_level():
    F = stats.norm.cdf
    h = lambda z: 0.1 * stats.norm.pdf(z)  # a valid perturbation direction
    out = kosorok_ch2_quantile_hadamard_inequality(F, h, t_n=0.01, p=0.7)
    assert out["sandwich_holds"] is True
    assert out["lower"] <= out["upper"]
    # the perturbed quantile sits near the unperturbed one for small t
    assert out["xi_perturbed"] == pytest.approx(stats.norm.ppf(0.7), abs=0.05)
    with pytest.raises(ValueError):
        kosorok_ch2_quantile_hadamard_inequality(F, h, t_n=0.01, p=1.5)
    with pytest.raises(ValueError):
        kosorok_ch2_quantile_hadamard_inequality(F, h, t_n=0.0, p=0.7)


def test_quantile_bounds_collapse_onto_the_hadamard_derivative():
    F = stats.norm.cdf
    h = lambda z: 0.1 * stats.norm.pdf(z)
    p = 0.6
    gaps = []
    for t in (0.05, 0.01, 0.002):
        out = kosorok_ch2_quantile_taylor_bounds(F, h, t_n=t, p=p)
        gaps.append(abs(out["gap"]))
        # the implied derivative is -h(xi_p)/f(xi_p), independent of t
        xi = stats.norm.ppf(p)
        assert out["implied_derivative"] == pytest.approx(
            -h(xi) / stats.norm.pdf(xi), rel=1e-3
        )
    # the bracket tightens as t shrinks
    assert gaps[-1] < gaps[0]
    with pytest.raises(ValueError):
        kosorok_ch2_quantile_taylor_bounds(F, h, t_n=-1.0, p=0.6)


def test_bootstrap_delta_method_centres_at_the_sample_not_the_truth():
    rng = np.random.default_rng(2)
    mu = 1.0
    X = rng.normal(mu, 1.0, 400)
    xbar = float(X.mean())
    boots = [float(rng.choice(X, size=400, replace=True).mean()) for _ in range(300)]
    out = kosorok_ch2_functional_delta_bootstrap(
        lambda z: z**2, np.array(xbar), [np.array(b) for b in boots],
        r_n=np.sqrt(400), mu=np.array(mu),
    )
    # correctly centred: mean near zero
    assert abs(out["mean"]) < 0.6
    # centring at the truth adds the original sampling error back in,
    # so it is BOTH biased and (weakly) more dispersed
    assert abs(out["truth_centred_mean"]) > abs(out["mean"])
    assert out["sd"] > 0
    with pytest.raises(ValueError):
        kosorok_ch2_functional_delta_bootstrap(lambda z: z, 1.0, [1.0], r_n=20.0)


def test_dqm_holds_for_a_normal_location_family():
    # N(theta, 1): score is (x - theta), DQM integral -> 0
    dens = lambda x, th: float(stats.norm.pdf(x, loc=th))
    score = lambda x: float(x)
    out = kosorok_ch3_differentiable_quadratic_mean(dens, score, theta=0.0)
    assert out["shrinking"] is True
    assert out["dqm_integrals"][-1] < 1e-3  # converging to zero
    assert out["score_mean"] == pytest.approx(0.0, abs=1e-8)  # score integrates to 0
    # a WRONG score does not give a vanishing DQM integral
    bad = kosorok_ch3_differentiable_quadratic_mean(dens, lambda x: 5.0, theta=0.0)
    assert bad["dqm_integrals"][-1] > out["dqm_integrals"][-1] * 100
    with pytest.raises(ValueError):
        kosorok_ch3_differentiable_quadratic_mean(dens, score, t_grid=[0.0])


def test_pathwise_derivative_checks_the_influence_function():
    rng = np.random.default_rng(3)
    n = 500
    x = rng.standard_normal(n)
    # influence function of the mean is (x - xbar): mean zero
    psi = x - x.mean()
    out = kosorok_ch3_pathwise_derivative(psi, x)
    assert out["mean_zero"] is True
    assert out["influence_var"] == pytest.approx(float(np.var(psi)), rel=1e-6)
    # the derivative along the score x is P[psi * x] ~ Var(x) = 1
    assert float(out["derivative"][0]) == pytest.approx(1.0, abs=0.2)
    # a non-centred candidate is flagged
    assert kosorok_ch3_pathwise_derivative(psi + 5.0, x)["mean_zero"] is False
    with pytest.raises(ValueError):
        kosorok_ch3_pathwise_derivative(psi, x[:10])


def test_efficient_influence_solves_or_reports_inconsistency():
    # a consistent full-rank system solves exactly
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    rhs = np.array([2.0, 8.0])
    out = kosorok_ch3_efficient_influence_general(A, rhs)
    assert out["chi"] == pytest.approx([1.0, 2.0])
    assert out["consistent"] is True
    assert out["efficient_variance"] == pytest.approx(5.0)
    assert out["rank"] == 2
    # a rank-deficient, inconsistent system means the parameter is NOT
    # pathwise differentiable -- reported, not silently least-squared
    Abad = np.array([[1.0, 1.0], [1.0, 1.0]])
    bad = kosorok_ch3_efficient_influence_general(Abad, np.array([1.0, 5.0]))
    assert bad["consistent"] is False
    assert bad["residual_norm"] > 1.0
    assert bad["rank"] == 1
    with pytest.raises(ValueError):
        kosorok_ch3_efficient_influence_general(A, np.array([1.0, 2.0, 3.0]))
