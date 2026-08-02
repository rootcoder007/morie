# morie.fn -- test file (rootcoder007/morie)
"""Fauzi and Maesono (2023), *Statistical Inference Based on Kernel
Distribution Function Estimators*.

The book has one organising problem -- the boundary -- and these
tests are built to make that problem visible rather than to check
that functions return floats. The naive estimators are asserted to
FAIL at the boundary and the Ch. 4 constructions are asserted to
succeed there, on the same data, so the comparison is the test.

Where a quantity has a closed form (the MRL of an exponential is its
mean, at every t; the optimal bandwidth of a second-order kernel;
the vanishing moments of an order-m kernel) the closed form is the
oracle. Nothing here is anchored to output the implementation
happened to produce.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.fzkde import fauzi_kde
from morie.fn.fzmise import fauzi_mise
from morie.fn.fzgkde import fauzi_gamma_kde
from morie.fn.fzkdfe import fauzi_kdfe
from morie.fn.fzbfkd import fauzi_boundary_free_kde
from morie.fn.fzcs1 import fauzi_cumulative_survival_1
from morie.fn.fzcs2 import fauzi_cumulative_survival_2
from morie.fn.fzb1t import fauzi_b1_coefficient
from morie.fn.fzb2t import fauzi_b2_coefficient
from morie.fn.fzb3t import fauzi_b3_coefficient
from morie.fn.fzmrln import fauzi_mrl_naive
from morie.fn.fzmr2 import fauzi_mrl_boundary_free_2
from morie.fn.fzt43 import fauzi_theorem_4_3
from morie.fn.fzt44 import fauzi_theorem_4_4
from morie.fn.fzt45 import fauzi_theorem_4_5
from morie.fn.fzt46 import fauzi_theorem_4_6
from morie.fn.fzc1c6 import fauzi_conditions_c1_c6
from morie.fn.fzkqe import fauzi_kernel_quantile
from morie.fn.fzamse import fauzi_quantile_amse
from morie.fn.fzkoc import fauzi_order_m_kernel
from morie.fn.fzmkrn import fauzi_muller_kernel
from morie.fn.fzl31 import fauzi_lemma_3_1


def exp_sample(n=600, rate=1.0, seed=0):
    """Exponential(rate) draws.

    Exponential is the right fixture for this book: its support is
    bounded below at zero (so the boundary problem is live), its
    density at the boundary is NOT zero (so the problem actually
    bites -- with a density vanishing at the edge the naive
    estimator would look fine), and its mean residual life is
    constant and equal to 1/rate at every t, which gives an exact
    oracle for Ch. 4.
    """
    return np.random.default_rng(seed).exponential(1.0 / rate, n)


# --------------------------------------------------------------- Ch. 1


def test_kde_integrates_to_one_and_admits_its_boundary_failure():
    x = exp_sample()
    o = fauzi_kde(x, grid=np.linspace(-1, 8, 400))
    assert abs(o["mass"] - 1.0) < 0.02
    assert o["boundary_consistent"] is False
    assert o["interior_bias_order"] == "O(h^2)"
    # the density is exp(-x) for x > 0 and 0 for x < 0. A symmetric
    # kernel cannot know that: it leaks mass onto the negative
    # half-line, where the truth is exactly zero.
    g, d = o["grid"], o["density"]
    assert np.trapezoid(d[g < 0], g[g < 0]) > 0.01


def test_kde_at_zero_is_biased_toward_one_half():
    """The signature boundary failure: at the edge of the support a
    symmetric kernel sees only half its neighbourhood, so it returns
    roughly half the true density. f(0) = 1 for a unit exponential;
    the estimate should sit near 0.5, and that is a defect, not
    noise -- it does not shrink as n grows."""
    est = [fauzi_kde(exp_sample(n, seed=s), grid=[0.0], h=n ** (-0.2))["density"][0]
           for n, s in ((400, 1), (4000, 2), (20000, 3))]
    for e in est:
        assert 0.35 < e < 0.72
    # more data does not repair it
    assert abs(est[-1] - 0.5) < 0.25


def test_mise_optimal_bandwidth_is_the_analytic_minimiser():
    o = fauzi_mise(1000, R_K=1 / (2 * np.sqrt(np.pi)), mu2_K=1.0, R_f2=1.0)
    rk = 1 / (2 * np.sqrt(np.pi))
    assert o["h_optimal"] == pytest.approx((rk / (1000 * 1.0 * 1.0)) ** 0.2, rel=1e-12)
    assert o["rate_exponent"] == -0.8
    assert o["parametric_rate_exponent"] == -1.0
    # h_opt is a true minimum: perturbing it either way costs MISE
    for f in (0.7, 1.4):
        assert fauzi_mise(1000, h=o["h_optimal"] * f, R_K=rk)["mise"] > o["mise_optimal"]
    # the two parts of MISE pull opposite ways -- that is the whole
    # bias-variance trade, and it must be visible in the numbers
    small = fauzi_mise(1000, h=0.05, R_K=rk)
    large = fauzi_mise(1000, h=0.5, R_K=rk)
    assert small["variance_part"] > large["variance_part"]
    assert small["bias_part"] < large["bias_part"]


def test_mise_rate_beats_no_smoothing_but_never_reaches_parametric():
    prev = None
    for n in (100, 1000, 10000, 100000):
        m = fauzi_mise(n, R_K=1 / (2 * np.sqrt(np.pi)))["mise_optimal"]
        if prev is not None:
            assert m < prev
        prev = m
    # n^{-4/5} is the ceiling for a second-order kernel: the ratio to
    # the parametric n^{-1} rate diverges rather than settling
    r = [fauzi_mise(n, R_K=0.28)["mise_optimal"] * n for n in (1e3, 1e6, 1e9)]
    assert r[0] < r[1] < r[2]


def test_gamma_kernel_is_consistent_at_zero_where_the_gaussian_is_not():
    """Chen's gamma kernel has support [0, inf) -- exactly the
    support of the data -- so no mass is ever placed where the
    density is zero, and the estimator converges at the boundary."""
    x = exp_sample(4000, seed=7)
    h = 4000 ** (-0.4)
    gam = fauzi_gamma_kde(x, grid=[0.0], h=h)["density"][0]
    gau = fauzi_kde(x, grid=[0.0], h=4000 ** (-0.2))["density"][0]
    # truth is f(0) = 1
    assert abs(gam - 1.0) < abs(gau - 1.0)
    assert abs(gam - 1.0) < 0.25


def test_gamma_kde_is_a_density_and_never_negative():
    x = exp_sample(1500, seed=11)
    o = fauzi_gamma_kde(x, grid=np.linspace(0, 10, 500), h=0.05)
    assert np.all(o["density"] >= 0)
    assert abs(o["mass"] - 1.0) < 0.05
    assert o["boundary_consistent"] is True
    with pytest.raises(ValueError, match="0, infinity"):
        fauzi_gamma_kde(np.array([-1.0, 2.0]), grid=[1.0], h=0.1)


def test_modified_gamma_kernel_reports_the_faster_bias_order():
    x = exp_sample(1000, seed=13)
    plain = fauzi_gamma_kde(x, grid=[1.0], h=0.05)
    mod = fauzi_gamma_kde(x, grid=[1.0], h=0.05, modified=True)
    assert plain["bias_order"] != mod["bias_order"]
    assert mod["modified"] is True


# --------------------------------------------------------------- Ch. 2


def test_kdfe_is_continuous_monotone_and_close_to_the_edf():
    x = exp_sample(800, seed=17)
    g = np.linspace(0.1, 6, 200)
    o = fauzi_kdfe(x, grid=g)
    assert o["monotone"] is True
    assert o["uses_integrated_kernel"] is True
    assert np.all(o["F_hat"] >= -1e-12) and np.all(o["F_hat"] <= 1 + 1e-12)
    # it smooths the edf rather than replacing it
    assert np.max(np.abs(o["F_hat"] - o["F_empirical"])) < 0.05
    # and it tracks the truth 1 - exp(-t)
    assert np.max(np.abs(o["F_hat"] - (1 - np.exp(-g)))) < 0.06


def test_kdfe_bias_carries_f_prime_not_f_double_prime():
    """The distinction is the point of (2.2): smoothing a
    distribution function with W = int K gives bias
    h^2 mu_2 f'(x)/2, one derivative lower than the density
    estimator's f''. The documented term must say so."""
    o = fauzi_kdfe(exp_sample(200, seed=19), grid=[1.0])
    assert "f'" in o["bias_term"] or "fPRIME" in o["bias_term"].replace(" ", "")
    assert "f''" not in o["bias_term"]


def test_kdfe_beats_the_edf_where_the_edf_jumps():
    """A step function has no derivative and, between order
    statistics, is flat -- so its error at a point strictly inside a
    gap is bounded below by the gap. Smoothing removes exactly that
    error, which is why the KDFE exists."""
    rng = np.random.default_rng(23)
    err_s, err_e = [], []
    for s in range(12):
        x = rng.exponential(1.0, 120)
        g = np.linspace(0.2, 4, 60)
        o = fauzi_kdfe(x, grid=g)
        truth = 1 - np.exp(-g)
        err_s.append(np.mean((o["F_hat"] - truth) ** 2))
        err_e.append(np.mean((o["F_empirical"] - truth) ** 2))
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
        sig = min(sd, iqr) if iqr > 0 else sd
        assert o["bandwidth"] == pytest.approx(
            4 ** (1 / 3) * sig * 120 ** (-1 / 3), rel=1e-12)
    # this comparison is the reason the bandwidth rule matters: under
    # the n^{-1/5} density rule the estimator OVERSMOOTHS and loses to
    # the step function it is supposed to improve on
    assert np.mean(err_s) < np.mean(err_e)


def test_the_density_bandwidth_rule_would_lose_to_the_empirical_df():
    """The concrete cost of getting the rate wrong, measured rather
    than asserted. h_opt for a distribution function is a CUBE root:
    (2.3)-(2.4) put the bandwidth in the variance at O(h/n), with a
    negative sign, not at O(1/(nh)). Substituting the density rule
    n^{-1/5} oversmooths, and the smoothed estimate then has a LARGER
    mean squared error than the raw step function."""
    rng = np.random.default_rng(101)
    right, wrong, edf = [], [], []
    for _ in range(12):
        x = rng.exponential(1.0, 200)
        g = np.linspace(0.2, 4, 60)
        truth = 1 - np.exp(-g)
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
        sig = min(sd, iqr) if iqr > 0 else sd
        o = fauzi_kdfe(x, grid=g)
        w = fauzi_kdfe(x, grid=g, h=1.06 * sig * 200 ** (-1 / 5))
        right.append(np.mean((o["F_hat"] - truth) ** 2))
        wrong.append(np.mean((w["F_hat"] - truth) ** 2))
        edf.append(np.mean((o["F_empirical"] - truth) ** 2))
    assert np.mean(right) < np.mean(edf)     # the estimator earns its keep
    assert np.mean(wrong) > np.mean(edf)     # under the density rule it does not
    assert np.mean(right) < np.mean(wrong)


def test_the_two_bandwidth_rates_diverge_as_n_grows():
    """n^{-1/3} and n^{-1/5} are not a constant apart: the ratio
    grows like n^{2/15}, so the error compounds with sample size
    rather than washing out."""
    from morie.fn._fauzi import kdfe_bandwidth
    prev = None
    for n in (100, 10_000, 1_000_000):
        df = kdfe_bandwidth(None, sigma=1.0, n=n)
        density = 1.06 * 1.0 * n ** (-1 / 5)
        assert df == pytest.approx(4 ** (1 / 3) * n ** (-1 / 3), rel=1e-12)
        r = density / df
        if prev is not None:
            assert r > prev
        prev = r
    # the ratio is exactly (1.06 / 4^(1/3)) n^(2/15): slow, but it
    # never stops growing, and by n = 10^6 the density rule is more
    # than four times too wide
    assert prev == pytest.approx(
        1.06 / 4 ** (1 / 3) * 1_000_000 ** (2 / 15), rel=1e-12)
    assert prev > 4


# --------------------------------------------------------------- Ch. 4


def test_boundary_free_kde_carries_the_jacobian():
    """f~_X(t) = 1/(n h g'(g^-1(t))) sum K((g^-1(t) - g^-1(X_i))/h).

    Without the 1/g'(g^-1(t)) factor the result is not a density on
    the original scale at all -- it will not integrate to one. This
    test would pass on an implementation that dropped the factor only
    if the transformation were the identity, which it is not."""
    x = exp_sample(3000, seed=29)
    o = fauzi_boundary_free_kde(x, grid=np.linspace(0.01, 12, 600))
    assert abs(o["mass"] - 1.0) < 0.06
    # for the log transform g = exp, so g'(g^-1(t)) = t and the
    # change-of-variables factor is 1/t
    g = o["grid"]
    assert np.allclose(o["g_prime"], g, rtol=1e-10)
    assert np.allclose(o["jacobian"], 1.0 / g, rtol=1e-10)
    # the discriminating check is recovery of a KNOWN density on the
    # original scale. Lognormal data is exactly normal after the log
    # transform, so the transformed-scale estimate is as good as a
    # kernel estimate ever gets and any error left on the original
    # scale is the change of variables. Without the 1/t factor the
    # curve is off by a factor of t and misses badly.
    ln = np.random.default_rng(97).lognormal(0.0, 1.0, 4000)
    tt = np.linspace(0.2, 6.0, 300)
    est = fauzi_boundary_free_kde(ln, grid=tt)
    truth = np.exp(-(np.log(tt) ** 2) / 2) / (tt * np.sqrt(2 * np.pi))
    assert np.max(np.abs(est["density"] - truth)) < 0.05
    assert np.max(np.abs(est["density"] * est["g_prime"] - truth)) > 0.2


def test_boundary_free_kde_fixes_the_zero_bias_the_naive_one_has():
    x = exp_sample(4000, seed=31)
    t = 0.02
    bf = fauzi_boundary_free_kde(x, grid=[t])["density"][0]
    nv = fauzi_kde(x, grid=[t])["density"][0]
    assert abs(bf - 1.0) < abs(nv - 1.0)


def test_first_cumulative_survival_preserves_the_derivative_relation():
    """(4.8) is built so that d/dt of the cumulative survival is
    exactly minus the survival estimator; (4.17) is not. Both are
    valid estimators -- they differ in which structural identity they
    keep, and the flags must not claim otherwise."""
    x = exp_sample(800, seed=37)
    tg = np.linspace(0.2, 4, 300)
    o1 = fauzi_cumulative_survival_1(x, tg)
    o2 = fauzi_cumulative_survival_2(x, tg)
    assert o1["preserves_derivative_relation"] is True
    assert o2["preserves_derivative_relation"] is False
    d = np.gradient(o1["S_cumulative"], tg)
    assert np.max(np.abs(d + o1["S_survival"])) < 0.02
    assert o1["bias_coefficient"].startswith("b_2")
    assert o2["bias_coefficient"].startswith("b_3")
    assert o2["same_covariance_as_first"] is True


def test_cumulative_survival_tracks_the_exponential_truth():
    """For Exp(1), S(t) = e^-t and int_t^inf S = e^-t as well."""
    x = exp_sample(3000, seed=41)
    tg = np.linspace(0.3, 3.5, 120)
    for f in (fauzi_cumulative_survival_1, fauzi_cumulative_survival_2):
        o = f(x, tg)
        assert np.max(np.abs(o["S_survival"] - np.exp(-tg))) < 0.06
        assert np.max(np.abs(o["S_cumulative"] - np.exp(-tg))) < 0.08


def test_bias_coefficients_use_the_transformation_derivatives():
    """b_1, b_2, b_3 (4.14, 4.15, 4.21) are the reason the Ch. 4
    bias is computable. For g = exp, g' = g'' = g, so at
    z = g^-1(t) = log t both derivatives equal t -- a check the
    implementation cannot pass by accident."""
    t = 2.5
    for f in (fauzi_b1_coefficient, fauzi_b2_coefficient, fauzi_b3_coefficient):
        o = f(t, f_X=0.5, f_X_prime=-0.5, S_X=0.3)
        assert o["g_prime"] == pytest.approx(t, rel=1e-12)
        assert o["g_double_prime"] == pytest.approx(t, rel=1e-12)
        assert o["bias_order"].startswith("O(h^2)")
        assert np.isfinite(o["b_1" if f is fauzi_b1_coefficient else
                             ("b_2" if f is fauzi_b2_coefficient else "b_3")])
    # under the identity transform g'' = 0, which must change the answer
    a = fauzi_b1_coefficient(t, f_X=0.5, f_X_prime=-0.5, S_X=0.3)
    b = fauzi_b1_coefficient(t, f_X=0.5, f_X_prime=-0.5, S_X=0.3,
                             transform="identity")
    assert b["g_double_prime"] == 0.0
    assert a["b_1"] != pytest.approx(b["b_1"])


def test_mean_residual_life_of_an_exponential_is_its_mean_everywhere():
    """The memoryless property: E[X - t | X > t] = 1/lambda for
    every t. A constant oracle at every grid point, which is a much
    stronger test than a single value."""
    x = exp_sample(4000, rate=0.5, seed=43)   # mean 2
    tg = np.linspace(0.5, 3.0, 40)
    o = fauzi_mrl_boundary_free_2(x, tg)
    assert np.max(np.abs(o["mrl"] - 2.0)) < 0.35
    assert o["bias_order"].startswith("O(h^2)")
    assert o["variance_vanishes_at_boundary"] is True


def test_boundary_free_mrl_beats_the_naive_one_at_the_boundary():
    """This is the book's headline claim, and it is only worth
    anything if measured where it matters. Near t = 0 the naive
    (4.2) estimator's bias is O(h) and can degrade to O(1); the
    Ch. 4 estimator stays O(h^2)."""
    rng = np.random.default_rng(47)
    tg = np.array([0.01, 0.05, 0.1])
    e_nv, e_bf = [], []
    for _ in range(10):
        x = rng.exponential(1.0, 1500)
        e_nv.append(np.mean(np.abs(fauzi_mrl_naive(x, tg)["mrl"] - 1.0)))
        e_bf.append(np.mean(np.abs(fauzi_mrl_boundary_free_2(x, tg)["mrl"] - 1.0)))
    assert np.mean(e_bf) < np.mean(e_nv)
    assert fauzi_mrl_naive(exp_sample(200), tg)["boundary_safe"] is False


def test_theorem_4_3_biases_differ_only_in_b2_versus_b3():
    """(4.25)-(4.28): Bias[m~_{X,1}] uses b_2, Bias[m~_{X,2}] uses
    b_3, and everything else in the two expressions is identical.
    Holding b_2 = b_3 must therefore collapse them onto each other."""
    kw = dict(t=1.0, S_X=0.4, S_bar_X=0.5, m_X=1.25, b1=0.3, n=500, h=0.1)
    same = fauzi_theorem_4_3(b2=0.7, b3=0.7, **kw)
    assert same["bias_1"] == pytest.approx(same["bias_2"], rel=1e-12)
    diff = fauzi_theorem_4_3(b2=0.7, b3=-0.2, **kw)
    assert diff["bias_1"] != pytest.approx(diff["bias_2"])
    assert diff["bias_1"] == pytest.approx(same["bias_1"], rel=1e-12)
    # bias is O(h^2): quartering h divides it by sixteen
    a = fauzi_theorem_4_3(b2=0.7, b3=-0.2, **{**kw, "h": 0.4})["bias_1"]
    b = fauzi_theorem_4_3(b2=0.7, b3=-0.2, **{**kw, "h": 0.1})["bias_1"]
    assert a / b == pytest.approx(16.0, rel=1e-9)
    # variance is O(1/n) -- the bandwidth is a lower-order effect
    v = [fauzi_theorem_4_3(b2=0.7, b3=-0.2, **{**kw, "n": n})["variance"]
         for n in (500, 5000)]
    assert v[0] / v[1] == pytest.approx(10.0, rel=0.05)


def test_theorem_4_4_is_a_standard_normal_statement():
    o = fauzi_theorem_4_4(mrl_hat=1.2, mrl_true=1.0, variance=0.01)
    assert o["z"] == pytest.approx(2.0, rel=1e-12)
    assert o["p_two_sided"] == pytest.approx(0.04550026389635842, rel=1e-9)
    assert o["valid_at_boundary"] is True
    # an estimate at the truth cannot be evidence against it
    assert fauzi_theorem_4_4(1.0, 1.0, 0.01)["p_two_sided"] == pytest.approx(1.0)


def test_theorem_4_5_reports_a_supremum_and_where_it_is_attained():
    tg = np.linspace(0, 5, 501)
    err = 0.3 * np.exp(-((tg - 3.0) ** 2) / 0.02)
    o = fauzi_theorem_4_5(mrl_hat=1.0 + err, mrl_true=np.ones_like(tg), t_grid=tg)
    assert o["sup_error"] == pytest.approx(0.3, rel=1e-9)
    assert o["argmax_t"] == pytest.approx(3.0, abs=0.02)
    assert o["mode"].startswith("uniform")
    # restricting the interval must exclude the spike
    r = fauzi_theorem_4_5(1.0 + err, np.ones_like(tg), tg, interval=(0.0, 2.0))
    assert r["sup_error"] < 1e-6


def test_theorem_4_6_recovers_the_sample_mean_at_the_start_of_support():
    """(4.29): m~(a_1) + a_1 = Xbar + O_p(h^2). At the left end of
    the support nobody has failed yet, so the expected remaining
    lifetime is the expected lifetime."""
    x = exp_sample(2000, seed=53)
    o = fauzi_theorem_4_6(x, a1=0.0, mrl_at_a1=fauzi_mrl_boundary_free_2(
        x, [1e-6])["mrl"][0])
    assert o["identity_lhs"] == pytest.approx(o["sample_mean"], abs=0.2)
    assert o["expected_order"] == "O(h^2)"
    assert o["gap"] >= 0


def test_conditions_c1_c6_name_which_ones_bind():
    o = fauzi_conditions_c1_c6(exp_sample(300, seed=59))
    assert o["C3_bijective"] is True
    assert set(o["binding_in_practice"]) == {"C5", "C6"}
    assert len(o["conditions"]) == 6


# --------------------------------------------------------------- Ch. 3


def test_kernel_quantile_smooths_in_p_and_sits_near_the_sample_quantile():
    x = exp_sample(2000, seed=61)
    for p in (0.25, 0.5, 0.75):
        o = fauzi_kernel_quantile(x, p)
        assert o["quantile"][0] == pytest.approx(
            o["sample_quantile"][0], abs=0.15)
        # truth: Q(p) = -log(1-p)
        assert o["quantile"][0] == pytest.approx(-np.log(1 - p), abs=0.15)
        assert o["weights_sum"][0] == pytest.approx(1.0, abs=0.02)
    assert "PROBABILITY" in o["smooths_in"].upper()
    # a vector of levels must give the same answers as the scalar calls
    v = fauzi_kernel_quantile(x, [0.25, 0.5, 0.75])["quantile"]
    assert v == pytest.approx(
        [fauzi_kernel_quantile(x, p)["quantile"][0]
         for p in (0.25, 0.5, 0.75)], rel=1e-12)


def test_kernel_quantile_is_continuous_in_p_where_the_sample_one_jumps():
    """The sample quantile is a step function of p -- it uses one
    order statistic and hops to the next. The kernel estimator is a
    weighted sum of ALL of them, so it moves smoothly."""
    x = exp_sample(60, seed=67)
    ps = np.linspace(0.3, 0.7, 400)
    kq = np.array([fauzi_kernel_quantile(x, p)["quantile"][0] for p in ps])
    sq = np.quantile(x, ps)
    assert np.max(np.abs(np.diff(kq))) < np.max(np.abs(np.diff(sq)))
    assert np.all(np.diff(kq) > -1e-9)   # still monotone in p


def test_quantile_amse_is_the_two_equivalent_forms_of_the_same_number():
    """(3.3): p(1-p)/(n f(Q(p))^2) and Q'(p)^2 p(1-p)/n are the same
    quantity, because Q' = 1/f(Q). Supplying either input must give
    the same answer."""
    p, n = 0.3, 500
    f = np.exp(np.log(1 - p))           # Exp(1): f(Q(p)) = 1 - p
    a = fauzi_quantile_amse(p, n, f_at_quantile=f)["amse"][0]
    b = fauzi_quantile_amse(p, n, Q_prime=1.0 / f)["amse"][0]
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(p * (1 - p) / (n * f ** 2), rel=1e-12)
    assert fauzi_quantile_amse(p, n, f_at_quantile=f)["se"][0] == \
        pytest.approx(np.sqrt(a), rel=1e-12)
    # halving n doubles the AMSE
    assert fauzi_quantile_amse(p, 250, f_at_quantile=f)["amse"][0] == \
        pytest.approx(2 * a, rel=1e-12)


def test_quantile_amse_blows_up_in_the_tail_despite_the_binomial_part():
    """p(1-p) shrinks toward the tails, which would suggest quantiles
    are EASIER to estimate there. They are not: 1/f^2 grows faster.
    The book flags this and so must the output."""
    n = 1000
    prev = None
    for p in (0.5, 0.9, 0.99, 0.999):
        f = 1 - p                       # Exp(1)
        a = fauzi_quantile_amse(p, n, f_at_quantile=f)
        assert a["binomial_part"][0] == pytest.approx(p * (1 - p) / n, rel=1e-12)
        if prev is not None:
            assert a["amse"][0] > prev
        prev = a["amse"][0]


def test_order_m_kernel_moments_vanish_up_to_m_minus_one():
    u = np.linspace(-12, 12, 40001)
    for m in (2, 4, 6):
        o = fauzi_order_m_kernel(u, m=m)
        K = o["K"]
        assert np.trapezoid(K, u) == pytest.approx(1.0, abs=1e-8)
        for j in range(1, m):
            assert abs(np.trapezoid(u ** j * K, u)) < 1e-6
        # the m-th moment is finite and NOT zero -- that is what
        # makes the order exactly m rather than higher
        assert abs(np.trapezoid(u ** m * K, u)) > 1e-6
        assert o["bias_order"] == f"O(h^{m})"
    with pytest.raises(ValueError, match="2, 4 or 6"):
        fauzi_order_m_kernel(u, m=3)


def test_higher_order_kernels_must_go_negative():
    """A non-negative function with a vanishing second moment does
    not exist. So the faster bias rate is bought with negative
    kernel values, and therefore possibly-negative density
    estimates. The flag must be honest about it."""
    u = np.linspace(-5, 5, 2001)
    assert fauzi_order_m_kernel(u, m=2)["takes_negative_values"] is False
    for m in (4, 6):
        assert fauzi_order_m_kernel(u, m=m)["takes_negative_values"] is True


def test_muller_kernel_turns_negative_exactly_at_root_three():
    """(3 - u^2)phi(u)/2 changes sign where u^2 = 3, by
    construction. An implementation with the wrong constant would
    still be smooth and still integrate to one, but the sign change
    would move."""
    o = fauzi_muller_kernel(np.array([0.0, 1.0, np.sqrt(3.0), 2.0]))
    assert o["negative_beyond"] == pytest.approx(np.sqrt(3.0), rel=1e-12)
    K = o["K"]
    assert K[0] > 0 and K[1] > 0
    assert abs(K[2]) < 1e-12
    assert K[3] < 0
    assert o["bias_order"] == "O(h^4)"
    u = np.linspace(-12, 12, 40001)
    Ku = fauzi_muller_kernel(u)["K"]
    assert np.trapezoid(Ku, u) == pytest.approx(1.0, abs=1e-8)
    assert abs(np.trapezoid(u ** 2 * Ku, u)) < 1e-6      # mu_2 = 0


def test_lemma_3_1_remainder_is_smaller_order_than_the_linear_term():
    """A Bahadur-type representation is only useful if the remainder
    really is negligible -- otherwise the asymptotic variance it
    licenses is wrong. Test it as the lemma states it: the remainder
    must shrink relative to the leading term as n grows."""
    ratios = []
    for n, s in ((250, 71), (2500, 73), (25000, 79)):
        o = fauzi_lemma_3_1(exp_sample(n, seed=s), 0.5, q_true=np.log(2.0))
        assert o["centre"] == pytest.approx(np.log(2.0), rel=1e-12)
        ratios.append(abs(o["remainder"]) / max(abs(o["linear_term"]), 1e-12))
        assert o["asymptotic_variance"] == pytest.approx(
            0.25 / (n * o["density_at_quantile"] ** 2), rel=1e-12)
    assert ratios[-1] < ratios[0]
    # centred on the sample quantile the linear term collapses, and
    # the module has to say so rather than quietly reporting a
    # decomposition that carries no information
    d = fauzi_lemma_3_1(exp_sample(2500, seed=73), 0.5)
    assert abs(d["linear_term"]) < 1e-2 * abs(
        fauzi_lemma_3_1(exp_sample(2500, seed=73), 0.5,
                        q_true=np.log(2.0))["linear_term"])
    assert "degenerate" in d["centred_at"]


def test_lemma_3_1_influence_function_is_centred():
    """The linear term is an i.i.d. average of influence-function
    values, so those values must average to (nearly) zero -- that is
    what makes the representation a mean-zero average at all."""
    o = fauzi_lemma_3_1(exp_sample(3000, seed=83), 0.4,
                        q_true=-np.log(0.6))
    assert abs(np.mean(o["influence"])) < 0.05
    assert o["linear_term"] == pytest.approx(np.mean(o["influence"]), rel=1e-9)
    assert "Bahadur" in o["representation"]
