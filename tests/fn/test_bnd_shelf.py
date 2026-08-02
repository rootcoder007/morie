# morie.fn -- test file (rootcoder007/morie)
"""The partial-identification shelf.

Partial identification is about what the data alone can say, so the
tests here are identities rather than approximations wherever the
literature states one: the Manski bound width is EXACTLY the support
width times the missing share, the no-assumption ATE interval always
contains zero, the Imbens-Manski critical value interpolates between
the one- and two-sided normal quantiles and hits both limits exactly,
the CHT criterion is exactly zero on the identified set, and the LP
bounds land on the vertices they must land on.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bndcvx import bound_convex_estimator
from morie.fn.bndest import bound_estimation
from morie.fn.bndnln import bound_nonlinear
from morie.fn.bndpl import bnp_density_pl
from morie.fn.bndvar import bound_variance_term


# --------------------------------------------------- Manski bounds


def test_manski_width_is_support_times_missing_share_exactly():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.random(n)
    obs = rng.random(n) < 0.8
    o = bound_estimation(y, obs, (0.0, 1.0))
    assert o["width"] == pytest.approx(1.0 * (1 - o["p_observed"]), rel=1e-12)
    assert o["lower"] <= float(y[obs].mean()) <= o["upper"]
    assert o["identified"] is False
    # fully observed data collapse the interval to the sample mean
    full = bound_estimation(y, np.ones(n, dtype=bool), (0.0, 1.0))
    assert full["identified"] is True
    assert full["lower"] == pytest.approx(full["upper"], rel=1e-12)
    assert full["lower"] == pytest.approx(float(y.mean()), rel=1e-12)


def test_manski_bounds_narrow_as_more_is_observed():
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.random(n)
    widths = []
    for p in (0.5, 0.7, 0.9):
        obs = rng.random(n) < p
        widths.append(bound_estimation(y, obs, (0.0, 1.0))["width"])
    assert widths[0] > widths[1] > widths[2]


def test_no_assumption_ate_bounds_always_contain_zero():
    """Manski (1990): the worst-case ATE interval has width exactly
    K1 - K0, so it can never sign an effect on its own. Bounds that
    exclude zero have smuggled in an assumption."""
    rng = np.random.default_rng(5)
    n = 2000
    T = (rng.random(n) < 0.5).astype(float)
    # a large, genuinely positive effect -- and the bounds still
    # must not sign it
    y = 0.6 * T + 0.4 * rng.random(n)
    o = bound_estimation(y, None, (0.0, 1.0), treatment=T)
    assert o["ate_width"] == pytest.approx(1.0, rel=1e-12)
    assert o["contains_zero"] is True
    assert o["ate_lower"] < 0 < o["ate_upper"]
    # the true effect is inside
    assert o["ate_lower"] <= 0.6 <= o["ate_upper"]


def test_manski_bounds_validate_the_support():
    y = np.array([0.5, 1.5, 0.2, 0.9])
    obs = np.array([True, True, True, True])
    with pytest.raises(ValueError, match="outside the declared support"):
        bound_estimation(y, obs, (0.0, 1.0))
    with pytest.raises(ValueError, match="K0 < K1"):
        bound_estimation(y, obs, (1.0, 0.0))
    with pytest.raises(ValueError, match="binary"):
        bound_estimation(y, None, (0.0, 2.0), treatment=y)


# ---------------------------------------------- Imbens-Manski CI


def test_imbens_manski_c_hits_both_limits_exactly():
    """c = z_{1-alpha} when the identified set is wide relative to
    noise, and z_{1-alpha/2} when it collapses to a point. The
    interpolation is the content of the construction and the limits
    are exact, not approximate."""
    from morie.fn import _stats_core as stats

    wide = bound_variance_term(0.2, 0.8, 1.0, 1.0, 400)
    point = bound_variance_term(0.5, 0.5, 1.0, 1.0, 400)
    assert wide["c"] == pytest.approx(stats.norm.ppf(0.95), rel=1e-9)
    assert point["c"] == pytest.approx(stats.norm.ppf(0.975), rel=1e-9)
    # strictly between the limits for an intermediate set
    mid = bound_variance_term(0.5, 0.55, 1.0, 1.0, 400)
    assert wide["c"] < mid["c"] < point["c"]
    # and c is monotone decreasing in the set's width
    cs = [bound_variance_term(0.5, 0.5 + d, 1.0, 1.0, 400)["c"]
          for d in (0.0, 0.02, 0.05, 0.2)]
    assert cs[0] > cs[1] > cs[2] > cs[3]


def test_imbens_manski_interval_covers_the_parameter():
    """Coverage of the TRUE PARAMETER at 95%, measured as a
    frequency. The parameter is set at the lower bound, the worst
    case for a parameter-covering interval."""
    rng = np.random.default_rng(7)
    n = 400
    theta = 0.3            # true parameter AT the lower bound
    width = 0.2
    hits = 0
    reps = 300
    for _ in range(reps):
        x = theta + rng.standard_normal(n)
        lo_hat = float(x.mean())
        hi_hat = lo_hat + width
        s = float(x.std(ddof=1))
        ci = bound_variance_term(lo_hat, hi_hat, s, s, n)["ci"]
        hits += int(ci[0] <= theta <= ci[1])
    assert hits / reps > 0.92


def test_imbens_manski_validates_inputs():
    with pytest.raises(ValueError, match="at least lower_hat"):
        bound_variance_term(0.8, 0.2, 1.0, 1.0, 100)
    with pytest.raises(ValueError, match="positive"):
        bound_variance_term(0.2, 0.8, 0.0, 1.0, 100)
    with pytest.raises(ValueError, match="alpha"):
        bound_variance_term(0.2, 0.8, 1.0, 1.0, 100, alpha=1.5)


# ------------------------------------------------ CHT criterion


def interval_outcome(n=800, theta=2.0, half=1.0, seed=5):
    """E[L] <= theta <= E[U] with L, U centred at theta -/+ half:
    the identified set is [theta - half, theta + half] exactly."""
    rng = np.random.default_rng(seed)
    L = theta - half + rng.normal(scale=0.3, size=n)
    U = theta + half + rng.normal(scale=0.3, size=n)
    return np.column_stack([L, U])


def g_interval(d, th):
    return np.column_stack([d[:, 0] - th, th - d[:, 1]])


def test_cht_criterion_is_zero_on_the_identified_set():
    """Deep inside the identified set every sample moment is
    negative, so with the positive-part criterion Q_n is EXACTLY
    zero -- not small, zero."""
    d = interval_outcome()
    o = bound_nonlinear(d, g_interval, np.linspace(0, 4, 81), B=200)
    grid = o["theta_grid"]
    inside = (grid > 1.2) & (grid < 2.8)
    assert np.all(o["criterion"][inside] == 0.0)
    # and strictly positive well outside
    outside = (grid < 0.5) | (grid > 3.5)
    assert np.all(o["criterion"][outside] > 0)


def test_cht_set_estimate_recovers_the_identified_set():
    d = interval_outcome(n=2000, seed=9)
    o = bound_nonlinear(d, g_interval, np.linspace(0, 4, 161), B=200)
    est = o["set_estimate"]
    assert est.min() == pytest.approx(1.0, abs=0.1)
    assert est.max() == pytest.approx(3.0, abs=0.1)
    # the confidence set contains the set estimate -- levels nest
    lo, hi = o["confidence_set_bounds"]
    assert lo <= est.min() and hi >= est.max()


def test_cht_confidence_set_covers_the_identified_set_boundary():
    """The hard point for coverage is the BOUNDARY of the identified
    set, where the inequality binds. theta = 1 (the lower endpoint)
    must be inside the confidence region most of the time."""
    hits = 0
    reps = 60
    for s in range(reps):
        d = interval_outcome(n=400, seed=100 + s)
        o = bound_nonlinear(d, g_interval, np.array([1.0]), B=200, seed=s)
        hits += int(o["in_confidence_set"][0])
    assert hits / reps > 0.85


def test_cht_validates_inputs():
    d = interval_outcome(n=50)
    with pytest.raises(ValueError, match="alpha"):
        bound_nonlinear(d, g_interval, [1.0], alpha=2.0)
    with pytest.raises(ValueError, match="at least 10"):
        bound_nonlinear(d[:5], g_interval, [1.0])


# --------------------------------------------------- LP bounds


def test_lp_bounds_land_on_the_right_vertices():
    """min and max of x1 + x2 subject to x1 + 2 x2 = 1 on [0,1]^2:
    the minimum is 0.5 at (0, 0.5) and the maximum 1.0 at (1, 0).
    Exact vertices of a tiny polytope -- no tolerance games."""
    o = bound_convex_estimator([1.0, 1.0], A_eq=[[1.0, 2.0]], b_eq=[1.0])
    assert o["lower"] == pytest.approx(0.5, rel=1e-9)
    assert o["upper"] == pytest.approx(1.0, rel=1e-9)
    assert o["argmin"] == pytest.approx([0.0, 0.5], abs=1e-9)
    assert o["argmax"] == pytest.approx([1.0, 0.0], abs=1e-9)
    assert o["feasible"] is True and o["bounded"] is True
    assert o["sharp"] is True


def test_lp_bounds_tighten_as_restrictions_accumulate():
    """Adding an assumption can only shrink the identified set --
    monotonicity of identification in the maintained restrictions,
    which is the logic the MST framework runs on."""
    free = bound_convex_estimator([1.0, 1.0, 1.0])
    eq = bound_convex_estimator([1.0, 1.0, 1.0],
                                A_eq=[[1.0, 1.0, 0.0]], b_eq=[0.8])
    both = bound_convex_estimator([1.0, 1.0, 1.0],
                                  A_eq=[[1.0, 1.0, 0.0]], b_eq=[0.8],
                                  A_ub=[[0.0, 0.0, 1.0]], b_ub=[0.3])
    assert free["width"] >= eq["width"] >= both["width"]
    assert both["upper"] == pytest.approx(0.8 + 0.3, rel=1e-9)


def test_lp_infeasibility_is_a_specification_rejection():
    o = bound_convex_estimator([1.0], A_eq=[[1.0]], b_eq=[2.0],
                               bounds=[(0.0, 1.0)])
    assert o["feasible"] is False
    assert np.isnan(o["lower"]) and np.isnan(o["upper"])
    # and an unbounded direction is reported as such, not as a number
    u = bound_convex_estimator([1.0], bounds=[(0.0, None)])
    assert u["bounded"] is False or np.isinf(u["upper"])


def test_lp_validates_shapes():
    with pytest.raises(ValueError, match="inconsistent"):
        bound_convex_estimator([1.0, 1.0], A_eq=[[1.0]], b_eq=[1.0])
    with pytest.raises(ValueError, match="bounds has"):
        bound_convex_estimator([1.0, 1.0], bounds=[(0, 1)])


# --------------------------------------------------- Polya tree


def test_polya_tree_is_a_density_that_tracks_the_sample():
    rng = np.random.default_rng(11)
    x = rng.normal(size=1500)
    o = bnp_density_pl(x, grid=np.linspace(-4, 4, 400), tree_depth=7,
                       lo=-5, hi=5)
    assert o["mass"] == pytest.approx(1.0, abs=0.02)
    assert np.all(o["density"] >= 0)
    g = o["grid"]
    truth = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    assert np.mean(np.abs(o["density"] - truth)) < 0.05


def test_polya_tree_alpha_interpolates_base_and_histogram():
    """Large alpha smooths toward the uniform base measure; small
    alpha follows the data. Both directions must show."""
    rng = np.random.default_rng(13)
    x = rng.normal(loc=0.0, scale=0.5, size=800)
    g = np.linspace(-2, 2, 200)
    tight = bnp_density_pl(x, grid=g, alpha=0.01, lo=-3, hi=3)
    loose = bnp_density_pl(x, grid=g, alpha=1e4, lo=-3, hi=3)
    base = 1.0 / 6.0                      # uniform on [-3, 3]
    # huge alpha: essentially the base measure everywhere
    assert np.max(np.abs(loose["density"] - base)) < 0.02
    # small alpha: peaked where the data are
    assert tight["density"][100] > 3 * base
    assert bnp_density_pl(x, alpha=1.0)["alpha_rule"].startswith("alpha_m")


def test_polya_tree_validates_inputs():
    x = np.random.default_rng(17).normal(size=50)
    with pytest.raises(ValueError, match="tree_depth"):
        bnp_density_pl(x, tree_depth=0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        bnp_density_pl(x, alpha=-1.0)
    with pytest.raises(ValueError, match="at least 2"):
        bnp_density_pl(np.array([1.0]))
