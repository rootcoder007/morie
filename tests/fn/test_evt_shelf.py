# morie.fn -- test file (rootcoder007/morie)
"""The extreme-value shelf.

The oracles here are exact: a Pareto(alpha) tail has xi = 1/alpha, a
uniform tail xi = -1, an exponential tail xi = 0; the max-AR process
X_t = max(a X_{t-1}, (1-a) Z_t) has extremal index exactly 1 - a; the
GEV and GPD fixtures are simulated at known parameters and the
L-moment inversions must return them; and the Pickands dependence
function is 1 under independence and max(t, 1-t) under complete
dependence. Sign conventions are load-bearing (Hosking's k = -xi)
and tested explicitly.
"""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn._evt import gev_from_lmoments, l_moments, pwm_b
from morie.fn.evdedh import ev_dedh
from morie.fn.evextidx import ev_extremal_runs
from morie.fn.evextint import ev_extremal_intervals
from morie.fn.evextsl import ev_extremal_sliding
from morie.fn.evgevlm import ev_gev_lmoments
from morie.fn.evgevp2 import ev_gev_pwm
from morie.fn.evgpdpw import ev_gpd_pwm
from morie.fn.evhill import ev_hill
from morie.fn.evmadog import ev_madogram
from morie.fn.evpick import ev_pickands
from morie.fn.hillEst import hill_estimator


def max_ar(alpha, n, seed):
    """Max-autoregressive process with extremal index exactly 1 - alpha."""
    r = np.random.default_rng(seed)
    z = r.random(n) ** -1.0
    x = np.empty(n)
    x[0] = z[0]
    for t in range(1, n):
        x[t] = max(alpha * x[t - 1], (1 - alpha) * z[t])
    return x


# ------------------------------------------------- tail indices


def test_hill_recovers_the_pareto_tail_index():
    """Pareto(b = 4) has xi = 1/4 exactly, and for an EXACT Pareto
    the log-excesses are exactly exponential, so Hill is unbiased."""
    x = stats.pareto.rvs(b=4, size=50_000, random_state=5)
    o = ev_hill(x, k=500)
    assert o["xi"] == pytest.approx(0.25, abs=0.04)
    assert o["tail_alpha"] == pytest.approx(4.0, rel=0.2)
    assert o["se"] == pytest.approx(o["xi"] / np.sqrt(500), rel=1e-12)
    assert "xi > 0" in o["valid_for"]


def test_hill_returns_the_plot_when_k_is_omitted():
    """A single Hill number hides the k-instability; the plot is the
    honest output and its values must agree with per-k calls."""
    x = stats.pareto.rvs(b=3, size=5_000, random_state=7)
    o = ev_hill(x)
    ks = o["hill_plot_k"]
    xs = o["hill_plot_xi"]
    assert ks.size == xs.size > 50
    for k_probe in (10, 50, 200):
        i = int(np.flatnonzero(ks == k_probe)[0])
        assert xs[i] == pytest.approx(ev_hill(x, k=k_probe)["xi"], rel=1e-10)


def test_hill_refuses_a_non_positive_threshold():
    x = np.r_[stats.norm.rvs(size=200, random_state=9), -5.0]
    with pytest.raises(ValueError, match="not positive"):
        ev_hill(x - 10.0, k=50)


def test_hill_alias_shares_the_implementation():
    x = stats.pareto.rvs(b=4, size=2_000, random_state=11)
    a = hill_estimator(x, k=100)
    b = ev_hill(x, k=100)
    assert a["xi"] == pytest.approx(b["xi"], rel=1e-15)
    assert a["alias_of"] == "morie.fn.evhill.ev_hill"


def test_pickands_and_dedh_work_for_every_sign_of_xi():
    """The generality claim, tested at all three tail types. Hill
    cannot do the last two; these can."""
    heavy = stats.pareto.rvs(b=4, size=50_000, random_state=5)
    bounded = stats.uniform.rvs(size=50_000, random_state=7) + 1.0
    light = stats.expon.rvs(size=50_000, random_state=9) + 1.0
    assert ev_pickands(heavy, k=1500)["xi"] == pytest.approx(0.25, abs=0.08)
    assert ev_pickands(bounded, k=2000)["xi"] == pytest.approx(-1.0, abs=0.15)
    assert abs(ev_pickands(light, k=2000)["xi"]) < 0.15
    assert ev_dedh(heavy, k=2000)["xi"] == pytest.approx(0.25, abs=0.08)
    assert ev_dedh(bounded, k=2000)["xi"] == pytest.approx(-1.0, abs=0.1)
    assert abs(ev_dedh(light, k=1000)["xi"]) < 0.15


def test_dedh_first_term_is_the_hill_estimator():
    """DEdH Eq. (1.7): M1 IS Hill, and for heavy tails the correction
    goes to zero -- structure, not coincidence."""
    x = stats.pareto.rvs(b=4, size=50_000, random_state=5)
    d = ev_dedh(x, k=2000)
    h = ev_hill(x, k=2000)
    assert d["hill_part"] == pytest.approx(h["xi"], rel=1e-12)
    assert abs(d["correction"]) < 0.1
    assert "Hill" in d["agrees_with_hill_when"]


def test_pickands_pays_for_generality_in_variance():
    """Where both are valid (xi > 0), Hill's spread over replications
    is smaller -- the documented trade, measured."""
    h_est, p_est = [], []
    for s in range(40):
        x = stats.pareto.rvs(b=4, size=4_000, random_state=100 + s)
        h_est.append(ev_hill(x, k=200)["xi"])
        p_est.append(ev_pickands(x, k=200)["xi"])
    assert np.var(h_est) < 0.5 * np.var(p_est)


# ------------------------------------------------- GEV / GPD fits


def test_lmoments_and_pwms_agree_by_construction():
    x = stats.gumbel_r.rvs(size=500, random_state=13)
    l1, l2, l3, t3 = l_moments(x)
    assert l1 == pytest.approx(pwm_b(x, 0), rel=1e-12)
    assert l2 == pytest.approx(2 * pwm_b(x, 1) - pwm_b(x, 0), rel=1e-12)
    assert l3 == pytest.approx(
        6 * pwm_b(x, 2) - 6 * pwm_b(x, 1) + pwm_b(x, 0), rel=1e-10)


def test_gev_lmoment_fit_recovers_known_parameters():
    # scipy's genextreme c is MINUS xi, i.e. Hosking's k
    x = stats.genextreme.rvs(c=-0.3, loc=10, scale=2, size=20_000,
                             random_state=1)
    o = ev_gev_lmoments(x)
    assert o["mu"] == pytest.approx(10.0, abs=0.15)
    assert o["sigma"] == pytest.approx(2.0, abs=0.15)
    assert o["xi"] == pytest.approx(0.3, abs=0.05)
    assert o["k_hosking"] == pytest.approx(-o["xi"], rel=1e-12)
    assert o["tail_type"].startswith("Frechet")
    # the return level is the GEV quantile, checked against scipy
    rl = float(o["return_level_fn"](100))
    truth = float(stats.genextreme.ppf(0.99, c=-0.3, loc=10, scale=2))
    assert rl == pytest.approx(truth, rel=0.08)


def test_gev_gumbel_limit_and_weibull_branch():
    g = stats.gumbel_r.rvs(loc=5, scale=1.5, size=20_000, random_state=3)
    o = ev_gev_lmoments(g)
    assert abs(o["xi"]) < 0.05
    w = stats.genextreme.rvs(c=0.25, loc=0, scale=1, size=20_000,
                             random_state=5)
    ow = ev_gev_lmoments(w)
    assert ow["xi"] == pytest.approx(-0.25, abs=0.05)
    assert ow["tail_type"].startswith("Weibull")


def test_pwm_fit_is_the_lmoment_fit_exactly():
    """Hosking (1990): L-moments are linear combinations of the PWMs,
    so the 1985 PWM fit and the 1990 L-moment fit coincide. Exactly,
    not approximately."""
    x = stats.genextreme.rvs(c=-0.2, loc=3, scale=1, size=3_000,
                             random_state=7)
    a = ev_gev_pwm(x)
    b = ev_gev_lmoments(x)
    for key in ("mu", "sigma", "xi"):
        assert a[key] == pytest.approx(b[key], rel=1e-14)
    assert a["alias_of"] == "morie.fn.evgevlm.ev_gev_lmoments"


def test_gpd_pwm_recovers_parameters_and_forms_excesses():
    e = stats.genpareto.rvs(c=0.25, scale=1.5, size=20_000, random_state=2)
    o = ev_gpd_pwm(e)
    assert o["sigma"] == pytest.approx(1.5, abs=0.1)
    assert o["xi"] == pytest.approx(0.25, abs=0.05)
    assert o["reliable"] is True
    # threshold path: raw data in, excesses formed internally
    raw = 10.0 + e
    o2 = ev_gpd_pwm(raw, threshold=10.0)
    assert o2["xi"] == pytest.approx(o["xi"], rel=1e-12)
    assert o2["n_excesses"] == e.size
    with pytest.raises(ValueError, match="non-negative"):
        ev_gpd_pwm(np.array([-1.0, 2.0] * 10))


def test_gpd_flags_the_infinite_variance_regime():
    e = stats.genpareto.rvs(c=0.8, scale=1.0, size=20_000, random_state=4)
    o = ev_gpd_pwm(e)
    assert o["reliable"] is False
    assert o["reliability_note"] is not None


# ------------------------------------------------- extremal index


def test_the_extremal_index_of_max_ar_is_one_minus_alpha():
    """The exact oracle all three estimators must hit."""
    x = max_ar(0.5, 60_000, seed=1)
    u = float(np.quantile(x, 0.98))
    assert ev_extremal_runs(x, u)["theta"] == pytest.approx(0.5, abs=0.07)
    assert ev_extremal_intervals(x, u)["theta"] == pytest.approx(0.5,
                                                                abs=0.07)
    s = ev_extremal_sliding(x, block_length=200)
    assert s["theta"] == pytest.approx(0.5, abs=0.09)
    # a different alpha moves all three the right way
    x2 = max_ar(0.8, 60_000, seed=2)
    u2 = float(np.quantile(x2, 0.98))
    assert ev_extremal_intervals(x2, u2)["theta"] == pytest.approx(0.2,
                                                                   abs=0.07)


def test_independent_data_have_extremal_index_one():
    x = np.random.default_rng(3).random(60_000)
    u = float(np.quantile(x, 0.98))
    assert ev_extremal_intervals(x, u)["theta"] == pytest.approx(1.0,
                                                                 abs=0.05)
    assert ev_extremal_runs(x, u)["theta"] > 0.9
    s = ev_extremal_sliding(x, block_length=200)
    assert s["theta"] > 0.85


def test_theta_is_the_reciprocal_mean_cluster_size():
    x = max_ar(0.5, 40_000, seed=5)
    u = float(np.quantile(x, 0.98))
    o = ev_extremal_runs(x, u)
    assert o["mean_cluster_size"] == pytest.approx(1.0 / o["theta"],
                                                   rel=1e-12)
    assert o["n_clusters"] <= o["n_exceedances"]
    with pytest.raises(ValueError, match="lower the threshold"):
        ev_extremal_runs(x, float(x.max()) + 1.0)


def test_ferro_segers_uses_the_corrected_form_when_gaps_exceed_two():
    x = max_ar(0.5, 40_000, seed=7)
    u = float(np.quantile(x, 0.98))
    o = ev_extremal_intervals(x, u)
    assert o["form_used"].startswith("Eq. (34)")
    assert 0 < o["theta"] <= 1.0
    assert o["implied_mean_cluster_size"] == pytest.approx(
        1.0 / o["theta"], rel=1e-12)


def test_sliding_blocks_beat_disjoint_blocks_in_spread():
    """Northrop's reason to slide, measured over replications."""
    sl, dj = [], []
    for s in range(30):
        x = max_ar(0.5, 8_000, seed=200 + s)
        o = ev_extremal_sliding(x, block_length=90)
        sl.append(o["theta"])
        dj.append(o["theta_disjoint"])
    assert np.var(sl) < np.var(dj)


# ------------------------------------------------- madogram


def test_madogram_identifies_independence_and_complete_dependence():
    rng = np.random.default_rng(11)
    a = rng.random(4_000)
    b = rng.random(4_000)
    ind = ev_madogram(a, b)
    # independence: A = 1 everywhere, dependence summary 0
    assert np.all(ind["A"] > 0.93)
    assert ind["dependence_summary"] == pytest.approx(0.0, abs=0.06)
    com = ev_madogram(a, a)
    # complete dependence: A = max(t, 1-t), summary 1
    lower = np.maximum(ind["t"], 1 - ind["t"])
    assert com["A"] == pytest.approx(lower, abs=0.05)
    assert com["dependence_summary"] == pytest.approx(1.0, abs=0.1)


def test_madogram_estimate_respects_the_envelope():
    rng = np.random.default_rng(13)
    # a genuinely dependent extreme-value sample: logistic model via
    # componentwise maxima of common shocks
    z = rng.random((4_000, 2))
    common = rng.random(4_000)
    x = np.maximum(z[:, 0], common)
    y = np.maximum(z[:, 1], common)
    o = ev_madogram(x, y)
    lower = np.maximum(o["t"], 1 - o["t"])
    assert np.all(o["A"] >= lower - 1e-12)
    assert np.all(o["A"] <= 1 + 1e-12)
    assert 0.0 < o["dependence_summary"] < 1.0
    with pytest.raises(ValueError, match="strictly in"):
        ev_madogram(x, y, t=[0.0, 0.5])
