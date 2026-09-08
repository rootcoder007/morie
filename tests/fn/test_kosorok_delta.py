"""Kosorok tranche 3: delta method, Kaplan-Meier derivatives,
M-estimators, KMT, Ch. 1 models."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr022 import kosorok_ch1_multiplicative_intensity
from morie.fn.ksr025 import kosorok_ch1_penalized_loglikelihood
from morie.fn.ksr042 import kosorok_ch2_functional_delta_method
from morie.fn.ksr050 import kosorok_ch2_frechet_differentiability
from morie.fn.ksr051 import kosorok_ch2_continuous_invertibility
from morie.fn.ksr052 import kosorok_ch2_kaplan_meier_derivative
from morie.fn.ksr053 import kosorok_ch2_kaplan_meier_inverse
from morie.fn.ksr055 import kosorok_ch2_m_estimator_taylor_expansion
from morie.fn.ksr056 import kosorok_ch2_lad_lipschitz_bound
from morie.fn.ksr059 import kosorok_ch2_kmt_strong_approximation


def test_delta_method_remainder_vanishes_for_smooth_phi():
    # phi(x) = x^2 is smooth: the linearisation is exact to O(dev^2)
    out = kosorok_ch2_functional_delta_method(
        lambda x: x**2, np.array(2.01), np.array(2.0), r_n=100.0
    )
    assert out["derivative_converged"] is True
    # d/dx x^2 at 2 in direction 0.01 is 0.04
    assert float(out["derivative"]) == pytest.approx(0.04, abs=1e-6)
    assert abs(float(out["remainder"])) < 0.02  # the o_P(1) term
    # shrinking the deviation shrinks the remainder quadratically
    small = kosorok_ch2_functional_delta_method(
        lambda x: x**2, np.array(2.001), np.array(2.0), r_n=1000.0
    )
    assert abs(float(small["remainder"])) < abs(float(out["remainder"]))
    with pytest.raises(ValueError):
        kosorok_ch2_functional_delta_method(lambda x: x**2, 2.0, 2.0, r_n=0.0)


def test_frechet_check_separates_smooth_from_kinked_maps():
    hs = [np.array(0.1), np.array(0.05), np.array(0.01), np.array(0.001)]
    smooth = kosorok_ch2_frechet_differentiability(lambda th: th**2, 1.0, hs)
    assert smooth["ratio_shrinking"] is True
    assert smooth["ratios"][-1] < smooth["ratios"][0]
    # |.| at 0 is Hadamard (directionally) but NOT Frechet
    # differentiable. The default derivative is a single Jacobian, so
    # the ratio stays at 1 instead of vanishing -- a per-direction
    # derivative would wrongly report it as Frechet.
    kink = kosorok_ch2_frechet_differentiability(lambda th: abs(th), 0.0, hs)
    assert kink["ratios"].min() > 0.9  # ratio pinned at ~1, not shrinking
    assert kink["ratios"].max() > smooth["ratios"].max()
    with pytest.raises(ValueError):
        kosorok_ch2_frechet_differentiability(lambda th: th, 0.0, [np.array(0.1)])


def test_continuous_invertibility_detects_a_degenerate_map():
    # an invertible linear map has a positive lower Lipschitz constant
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    good = kosorok_ch2_continuous_invertibility(lambda th: A @ th, np.zeros(2))
    assert good["c_estimate"] >= 1.9  # the smallest singular value is 2
    assert good["is_upper_bound"] is True
    # a rank-deficient map collapses a direction, so c is ~0
    B = np.array([[1.0, 1.0], [1.0, 1.0]])
    bad = kosorok_ch2_continuous_invertibility(lambda th: B @ th, np.zeros(2))
    assert bad["c_estimate"] < 0.5
    assert bad["holds_for_c"] is None
    assert kosorok_ch2_continuous_invertibility(
        lambda th: A @ th, np.zeros(2), c=1.0
    )["holds_for_c"] is True
    with pytest.raises(ValueError):
        kosorok_ch2_continuous_invertibility(lambda th: th, np.zeros(2),
                                             theta_2=np.zeros(2))


def test_kaplan_meier_derivative_and_its_inverse_are_consistent():
    S0 = lambda u: np.exp(-0.5 * u)
    # L is an at-risk-type probability: positive at 0. A hazard-like
    # L(u) = 0.5u would vanish there and make the INVERSE integrand
    # 1/(L(u-)S_0(u-)) genuinely undefined, which ksr053 refuses.
    L = lambda u: np.exp(-0.3 * u)
    G = lambda u: u
    h = lambda u: 1.0
    d = kosorok_ch2_kaplan_meier_derivative(S0, L, G, h, 1.0)
    # both terms are negative contributions, so the derivative is < 0
    assert d["derivative"] < 0
    assert d["integral_term"] > 0
    assert d["boundary_term"] == pytest.approx(np.exp(-0.3))
    # vectorised over t, and monotone in magnitude
    dv = kosorok_ch2_kaplan_meier_derivative(S0, L, G, h, [0.5, 1.0, 2.0])
    assert dv["derivative"].size == 3
    assert abs(dv["derivative"][2]) > abs(dv["derivative"][0])
    # the inverse uses LEFT limits and returns finite values
    inv = kosorok_ch2_kaplan_meier_inverse(S0, L, lambda u: u, lambda u: u, 1.0)
    assert np.isfinite(inv["inverse"])
    assert inv["a_at_zero"] == 0.0
    with pytest.raises(ValueError):
        kosorok_ch2_kaplan_meier_derivative(S0, L, G, h, -1.0)
    with pytest.raises(ValueError):
        # a hazard-like L vanishing at 0 makes 1/(L(u-)S_0(u-)) blow
        # up; the module refuses rather than returning a finite lie
        kosorok_ch2_kaplan_meier_inverse(S0, lambda u: 0.5 * u, None,
                                         lambda u: u, 1.0)


def test_m_estimator_remainder_is_second_order_for_smooth_criteria():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(500)
    # squared loss: the expectation is exactly quadratic, so the
    # second-order remainder is exactly zero
    m = lambda th, x: (x - th[0]) ** 2
    thetas = [np.array([0.5]), np.array([0.2]), np.array([0.05]), np.array([0.01])]
    out = kosorok_ch2_m_estimator_taylor_expansion(m, thetas, np.array([0.0]), X)
    assert out["ratios"].max() < 1.5  # bounded ratio => O(||delta||^2)
    assert out["distances"][0] > out["distances"][-1]
    with pytest.raises(ValueError):
        kosorok_ch2_m_estimator_taylor_expansion(m, [np.array([0.1])],
                                                 np.array([0.0]), X)


def test_lad_lipschitz_bound_holds_pointwise():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((200, 3))
    y = rng.standard_normal(200)
    out = kosorok_ch2_lad_lipschitz_bound([1.0, 0.0, -1.0], [0.9, 0.2, -0.8], U, y)
    assert out["bound_holds"] is True
    assert out["max_ratio"] <= 1.0 + 1e-9
    # the bound is tight: a collinear perturbation attains ratio ~1
    Uc = np.ones((50, 1))
    tight = kosorok_ch2_lad_lipschitz_bound([0.0], [1.0], Uc, np.full(50, 10.0))
    assert tight["max_ratio"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        kosorok_ch2_lad_lipschitz_bound([1.0], [1.0, 2.0], U)


def test_kmt_refuses_to_invent_its_universal_constants():
    with pytest.raises(ValueError, match="universal"):
        kosorok_ch2_kmt_strong_approximation(1000, x=1.0)
    out = kosorok_ch2_kmt_strong_approximation(1000, x=2.0, a=1.0, b=1.0, c=1.0)
    assert out["threshold"] == pytest.approx((np.log(1000) + 2.0) / np.sqrt(1000))
    assert out["probability_bound"] == pytest.approx(np.exp(-2.0))
    # the threshold shrinks like log(n)/sqrt(n)
    big = kosorok_ch2_kmt_strong_approximation(100000, x=2.0, a=1.0, b=1.0, c=1.0)
    assert big["threshold"] < out["threshold"]
    with pytest.raises(ValueError):
        kosorok_ch2_kmt_strong_approximation(1, a=1.0, b=1.0, c=1.0)


def test_multiplicative_intensity_recovers_a_known_baseline():
    rng = np.random.default_rng(2)
    n = 600
    Z = rng.standard_normal((n, 1))
    beta = 0.7
    T = rng.exponential(1.0 / np.exp(Z[:, 0] * beta))
    C = rng.exponential(3.0, size=n)
    time, event = np.minimum(T, C), (T <= C).astype(float)
    out = kosorok_ch1_multiplicative_intensity(time, event, Z, beta=[beta],
                                               t=[0.25, 0.5, 1.0])
    # the true baseline is Lambda(t) = t for a unit exponential
    assert out["cumulative_hazard"][0] == pytest.approx(0.25, rel=0.35)
    assert out["cumulative_hazard"][2] == pytest.approx(1.0, rel=0.35)
    assert np.all(np.diff(out["cumulative_hazard"]) > 0)  # monotone
    assert out["expected_counts"].shape == (n, 3)
    with pytest.raises(ValueError):
        kosorok_ch1_multiplicative_intensity(time, np.zeros(n), Z)  # no events


def test_penalized_loglikelihood_squares_both_factors():
    ll = np.full(100, -1.5)
    out = kosorok_ch1_penalized_loglikelihood(ll, J_eta=2.0, lambda_n=0.5)
    assert out["mean_loglik"] == pytest.approx(-1.5)
    assert out["penalty"] == pytest.approx(0.25 * 4.0)  # lambda^2 * J^2
    assert out["criterion"] == pytest.approx(-1.5 - 1.0)
    # doubling lambda quadruples the penalty
    dbl = kosorok_ch1_penalized_loglikelihood(ll, J_eta=2.0, lambda_n=1.0)
    assert dbl["penalty"] == pytest.approx(4 * out["penalty"])
    assert kosorok_ch1_penalized_loglikelihood(ll, 2.0, 0.0)["penalty"] == 0.0
    with pytest.raises(ValueError):
        kosorok_ch1_penalized_loglikelihood(ll, J_eta=2.0, lambda_n=-1.0)
