# morie.fn -- test file (rootcoder007/morie)
"""The robust-regression and robust-scale shelf.

The organising trade-off is BREAKDOWN against EFFICIENCY, and the
tests are built around it: the calibration constants are recomputed
from their defining equations, the efficiency claims are measured as
variance ratios on clean normal data, and the breakdown claims are
measured by feeding each estimator the contamination it is supposed
to survive -- and the contamination it is documented NOT to survive.
Huber's failure under bad leverage is as much a test as MM's success.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from morie.fn._robust import (HUBER_C_95, QN_D, SN_C, TUKEY_C_95,
                              TUKEY_C_BREAKDOWN, mad_scale, s_scale,
                              tukey_rho)
from morie.fn.hubrr import huber_regression
from morie.fn.mestrg import m_regression
from morie.fn.mmestr import mm_estimator
from morie.fn.mmreg import mm_regression_estimator
from morie.fn.qnsc import qn_scale
from morie.fn.sensSlp import sens_slope
from morie.fn.sestrg import s_regression_estimator
from morie.fn.snsc import sn_scale
from morie.fn.taubrg import tau_regression
from morie.fn.theils import theil_sen


# ------------------------------------------------- the constants


def test_the_calibration_constants_solve_their_defining_equations():
    """None of 1.345, 1.5476, 4.685, 2.2191 is folklore; each solves
    a stated equation, and the equations are recomputed here."""
    # Huber c = 1.345: 95% efficiency at the normal
    c = HUBER_C_95
    num, _ = integrate.quad(stats.norm.pdf, -c, c)
    den, _ = integrate.quad(
        lambda u: np.clip(u, -c, c) ** 2 * stats.norm.pdf(u), -10, 10)
    assert num ** 2 / den == pytest.approx(0.95, abs=2e-4)
    # biweight c = 1.5476: E_Phi[rho] = 1/2, the 50%-breakdown calibration
    val, _ = integrate.quad(
        lambda u: tukey_rho(u, TUKEY_C_BREAKDOWN) * stats.norm.pdf(u),
        -10, 10)
    assert val == pytest.approx(0.5, abs=2e-4)
    # biweight c = 4.685: 95% efficiency
    c = TUKEY_C_95

    def psi(u):
        v = u / c
        return u * (1 - v ** 2) ** 2 if abs(v) < 1 else 0.0

    def dpsi(u):
        v = u / c
        return (1 - v ** 2) * (1 - 5 * v ** 2) if abs(v) < 1 else 0.0

    num, _ = integrate.quad(lambda u: dpsi(u) * stats.norm.pdf(u), -c, c)
    den, _ = integrate.quad(lambda u: psi(u) ** 2 * stats.norm.pdf(u), -c, c)
    assert num ** 2 / den == pytest.approx(0.95, abs=2e-4)
    # Qn d = 1/(sqrt(2) Phi^-1(5/8))
    assert QN_D == pytest.approx(1 / (np.sqrt(2) * stats.norm.ppf(5 / 8)),
                                 rel=1e-12)


# ------------------------------------------------- scales


def test_qn_and_sn_are_consistent_for_sigma_at_the_normal():
    rng = np.random.default_rng(3)
    for f in (qn_scale, sn_scale):
        vals = [f(rng.normal(scale=2.0, size=200))["value"]
                for _ in range(200)]
        assert np.mean(vals) == pytest.approx(2.0, rel=0.02)


def test_qn_and_sn_survive_forty_percent_contamination():
    """50% breakdown means 40% contamination barely moves them, while
    the standard deviation is destroyed."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=100)
    bad = np.r_[x[:60], np.full(40, 1000.0)]
    for f in (qn_scale, sn_scale):
        clean = f(x)["value"]
        dirty = f(bad)["value"]
        assert dirty < 3 * clean
        assert f(bad)["breakdown"] == 0.5
    assert np.std(bad) > 100 * np.std(x)


def test_qn_is_more_efficient_than_the_mad():
    """The paper's headline: Qn reaches 82% normal efficiency against
    the MAD's 37%. Measured as the variance ratio over replications."""
    rng = np.random.default_rng(7)
    qn_vals, mad_vals = [], []
    for _ in range(400):
        x = rng.normal(size=100)
        qn_vals.append(qn_scale(x)["value"])
        mad_vals.append(mad_scale(x))
    assert np.var(qn_vals) < 0.6 * np.var(mad_vals)


def test_sn_small_sample_correction_and_conventions():
    # without the finite-sample corrections both are biased low in
    # small samples; with them, the mean is near sigma even at n = 8
    rng = np.random.default_rng(9)
    vals = [sn_scale(rng.normal(size=8))["value"] for _ in range(3000)]
    assert np.mean(vals) == pytest.approx(1.0, rel=0.08)
    vals = [qn_scale(rng.normal(size=8))["value"] for _ in range(3000)]
    assert np.mean(vals) == pytest.approx(1.0, rel=0.08)
    with pytest.raises(ValueError, match="at least 2"):
        qn_scale([1.0])
    with pytest.raises(ValueError, match="at least 2"):
        sn_scale([1.0])


def test_the_m_scale_solves_its_defining_equation():
    rng = np.random.default_rng(11)
    r = rng.normal(scale=1.5, size=500)
    s = s_scale(r)
    assert float(np.mean(tukey_rho(r / s, TUKEY_C_BREAKDOWN))) == \
        pytest.approx(0.5, abs=1e-6)
    # and it is consistent for sigma at the normal
    assert s == pytest.approx(1.5, rel=0.1)


# ------------------------------------------------- regressions


def clean_line(n=200, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=n)
    return x, y


def test_huber_recovers_the_line_and_survives_vertical_outliers():
    x, y = clean_line()
    o = huber_regression(x, y)
    assert o["beta"] == pytest.approx([2.0, 3.0], abs=0.15)
    assert o["converged"] is True
    y2 = y.copy()
    y2[:40] += 30.0                       # 20% vertical outliers
    o2 = huber_regression(x, y2)
    assert o2["beta"][1] == pytest.approx(3.0, abs=0.2)
    # the outliers get downweighted, the clean points do not
    assert np.mean(o2["weights"][:40]) < 0.2
    assert np.mean(o2["weights"][40:]) > 0.9


def test_huber_breaks_under_bad_leverage_as_documented():
    """Huber's psi bounds the influence of a RESIDUAL, not of a
    design row; its regression breakdown point is 0, and the module
    says so. This test asserts the documented failure -- if Huber
    survived this design, the docstring would be wrong."""
    rng = np.random.default_rng(13)
    x, y = clean_line()
    x3, y3 = x.copy(), y.copy()
    x3[:60] = 8.0 + 0.1 * rng.normal(size=60)
    y3[:60] = -20.0
    o = huber_regression(x3, y3)
    assert abs(o["beta"][1] - 3.0) > 2.0
    assert o["breakdown"] == 0.0
    # and the estimators that claim to fix it, do
    for est in (mm_regression_estimator, tau_regression):
        r = est(x3, y3, seed=3)
        assert r["beta"] == pytest.approx([2.0, 3.0], abs=0.25)
        assert r["breakdown"] == 0.5


def test_s_estimator_has_high_breakdown_and_seeds_mm():
    rng = np.random.default_rng(17)
    x, y = clean_line(seed=17)
    x[:70] = 8.0 + 0.1 * rng.normal(size=70)   # 35% bad leverage
    y[:70] = -20.0
    s = s_regression_estimator(x, y, seed=5)
    assert s["beta"] == pytest.approx([2.0, 3.0], abs=0.3)
    assert s["gaussian_efficiency"] == pytest.approx(0.287)
    mm = mm_regression_estimator(x, y, seed=5)
    # the MM initial stage IS an S-estimate
    assert mm["beta_initial"] == pytest.approx(s["beta"], abs=0.3)
    assert mm["scale_held_fixed"] is True


def test_mm_is_more_efficient_than_s_on_clean_data():
    """The whole point of the M-step: same breakdown, much less
    variance at the clean model. Measured, not asserted."""
    rng = np.random.default_rng(19)
    s_err, mm_err = [], []
    for rep in range(60):
        x = rng.normal(size=80)
        y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=80)
        s_err.append(s_regression_estimator(x, y, n_subsets=100,
                                            seed=rep)["beta"][1] - 3.0)
        mm_err.append(mm_regression_estimator(x, y, n_subsets=100,
                                              seed=rep)["beta"][1] - 3.0)
    assert np.var(mm_err) < 0.7 * np.var(s_err)


def test_mm_estimator_alias_shares_the_implementation():
    x, y = clean_line(seed=23)
    a = mm_estimator(x, y, seed=1)
    b = mm_regression_estimator(x, y, seed=1)
    assert a["beta"] == pytest.approx(b["beta"], rel=1e-14)
    assert a["scale"] == pytest.approx(b["scale"], rel=1e-14)
    assert a["alias_of"] == "morie.fn.mmreg.mm_regression_estimator"


def test_m_regression_families_and_the_start_dependence_warning():
    x, y = clean_line(seed=29)
    h = m_regression(x, y, psi="huber")
    b = m_regression(x, y, psi="bisquare")
    assert h["monotone"] is True and h["unique_solution"] is True
    assert h["start_dependent_warning"] is None
    assert b["monotone"] is False
    assert "LOCAL" in b["start_dependent_warning"]
    assert h["beta"] == pytest.approx([2.0, 3.0], abs=0.15)
    assert b["beta"] == pytest.approx([2.0, 3.0], abs=0.15)
    with pytest.raises(ValueError, match="huber.*bisquare|'huber'"):
        m_regression(x, y, psi="cauchy")


def test_regressions_validate_their_inputs():
    with pytest.raises(ValueError, match="more observations"):
        huber_regression(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="positive"):
        huber_regression(np.arange(10.0), np.arange(10.0), c=-1)


# ------------------------------------------------- Theil-Sen


def test_theil_sen_recovers_the_slope_with_an_order_statistic_ci():
    x, y = clean_line(seed=31)
    o = theil_sen(x, y)
    assert o["slope"] == pytest.approx(3.0, abs=0.15)
    assert o["intercept"] == pytest.approx(2.0, abs=0.15)
    lo, hi = o["ci"]
    assert lo < 3.0 < hi
    assert o["breakdown"] == pytest.approx(1 - 1 / np.sqrt(2), rel=1e-12)
    assert "order statistics" in o["ci_method"]


def test_theil_sen_survives_29_percent_contamination():
    """Breakdown 1 - 1/sqrt(2): a quarter of the points corrupted
    moves the slope little; a clear majority of corrupted pairs
    (about 45% of points) breaks it. Both directions are the test."""
    rng = np.random.default_rng(37)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.3, size=n)
    y_bad = y.copy()
    y_bad[:50] = 50.0 + 10.0 * x[:50]          # 25% corrupted
    assert theil_sen(x, y_bad)["slope"] == pytest.approx(3.0, abs=0.4)
    # 45% corrupted is beyond the 29.3% breakdown, but breakdown is a
    # WORST-CASE statement: this particular contamination must be one
    # that actually attains it. Placing the bad points at extreme
    # leverage puts every mixed pair's slope near the bad line's too,
    # so about 70% of pairs vote for slope 10 -- and the median goes.
    x_w = x.copy()
    y_w = y.copy()
    x_w[:90] = 1000.0 + rng.normal(size=90)
    y_w[:90] = 50.0 + 10.0 * x_w[:90]
    assert abs(theil_sen(x_w, y_w)["slope"] - 3.0) > 3.0


def test_theil_sen_excludes_tied_x_pairs():
    """A tied pair's slope is undefined; Sen excludes it. Treating it
    as 0 or inf would bias the median, so the count is reported."""
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.2, 2.0, 3.0, 4.0])
    o = theil_sen(x, y)
    assert o["n_tied_x"] == 1
    assert o["n_pairs"] == 9               # C(5,2) - 1
    with pytest.raises(ValueError, match="tied"):
        theil_sen(np.ones(5), np.arange(5.0))


def test_sens_slope_is_theil_sen_on_the_time_index():
    rng = np.random.default_rng(41)
    y = 0.5 * np.arange(60) + rng.normal(size=60)
    o = sens_slope(y)
    t = np.arange(60.0)
    ref = theil_sen(t, y)
    assert o["slope"] == pytest.approx(ref["slope"], rel=1e-14)
    assert o["ci"] == pytest.approx(ref["ci"], rel=1e-14)
    assert o["trend"] == "increasing"
    assert o["alias_of"] == "morie.fn.theils.theil_sen"
    # a flat series shows no trend at this alpha
    flat = sens_slope(rng.normal(size=60))
    assert flat["trend"] == "no trend at this alpha"
    down = sens_slope(-0.5 * np.arange(60) + rng.normal(size=60))
    assert down["trend"] == "decreasing"


def test_sens_ci_covers_the_true_slope():
    """Sen's Sec. 5 interval is distribution-free; its coverage is a
    frequency and is measured as one, under non-normal noise where a
    residual-variance interval would be the wrong tool."""
    rng = np.random.default_rng(43)
    hits = 0
    reps = 200
    for _ in range(reps):
        t = np.arange(40.0)
        y = 1.0 + 0.3 * t + rng.standard_t(df=2, size=40)   # heavy tails
        lo, hi = sens_slope(y, t)["ci"]
        hits += int(lo <= 0.3 <= hi)
    assert hits / reps > 0.9
