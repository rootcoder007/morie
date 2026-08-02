# morie.fn -- test file (rootcoder007/morie)
"""The plug-in / resampling / Monte Carlo shelf.

These seven modules originally cited a textbook that is not in this
repository's reference library, by chapter, with no way to check the
chapters. Each has been re-grounded on a primary source that IS in
the library and was read from its PDF -- Silverman for the kernel
estimator and its window widths, MacKay for importance sampling, ESL
for the bootstrap variance and bagging, Kosorok for the functional
delta method and M-estimation -- except admissibility, which no text
in the library covers and which is therefore implemented from its
definition and says so.

The tests below check the things those sources actually assert,
including the places where the sources contradict the placeholder
docstrings.
"""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn._wsm import adaptive_spread, silverman_bandwidth
from morie.fn.wsmadm import wasserman_admissible
from morie.fn.wsmbgn import wasserman_bagging
from morie.fn.wsmboo import wasserman_bootstrap
from morie.fn.wsmiis import wasserman_importance_sampling
from morie.fn.wsmkdn import wasserman_kde
from morie.fn.wsmmle import wasserman_mle
from morie.fn.wsmpst import wasserman_plug_in_estimator


# ------------------------------------------------- Silverman, KDE


def test_silverman_rule_is_0_9_A_not_1_06_sigma():
    """Silverman (3.31) is h = 0.9 A n^{-1/5} with A = min(sd,
    IQR/1.34) from (3.30). The widely-copied 1.06 sigma n^{-1/5} is
    (3.28), the pure normal reference that the book presents as a
    starting point and then improves on twice."""
    x = np.random.default_rng(3).normal(size=500)
    a = adaptive_spread(x)
    assert silverman_bandwidth(x, "3.31") == pytest.approx(
        0.9 * a * 500 ** -0.2, rel=1e-12)
    assert silverman_bandwidth(x, "3.28") == pytest.approx(
        1.06 * np.std(x, ddof=1) * 500 ** -0.2, rel=1e-12)
    r = np.subtract(*np.percentile(x, [75, 25]))
    assert silverman_bandwidth(x, "3.29") == pytest.approx(
        0.79 * r * 500 ** -0.2, rel=1e-12)
    # on normal data A is the standard deviation, so the two rules
    # differ by exactly the constant 0.9 / 1.06
    assert a == pytest.approx(np.std(x, ddof=1), rel=0.05)
    assert (silverman_bandwidth(x, "3.31")
            < silverman_bandwidth(x, "3.28"))
    with pytest.raises(ValueError, match="3.28"):
        silverman_bandwidth(x, "silverman")


def test_the_adaptive_spread_resists_an_outlier_that_moves_the_sd():
    """(3.30) is the whole reason (3.31) beats (3.28) off the normal
    model: one contaminating point moves the standard deviation a
    long way and the interquartile range hardly at all, so the
    normal-reference rule oversmooths and the adaptive one does not."""
    x = np.random.default_rng(5).normal(size=400)
    dirty = np.r_[x, [60.0]]
    assert np.std(dirty, ddof=1) > 2 * np.std(x, ddof=1)
    assert adaptive_spread(dirty) == pytest.approx(adaptive_spread(x),
                                                   rel=0.15)
    clean_h = silverman_bandwidth(x, "3.31")
    dirty_h = silverman_bandwidth(dirty, "3.31")
    assert dirty_h == pytest.approx(clean_h, rel=0.15)
    # the normal reference is not nearly so stable
    assert (silverman_bandwidth(dirty, "3.28")
            > 2 * silverman_bandwidth(x, "3.28"))


def test_kde_is_a_density_and_matches_silverman_2_2a():
    x = np.random.default_rng(7).normal(size=600)
    g = np.linspace(-4, 4, 401)
    o = wasserman_kde(g, x, h=0.35)
    direct = np.array([
        np.mean(np.exp(-0.5 * ((t - x) / 0.35) ** 2)
                / (0.35 * np.sqrt(2 * np.pi))) for t in g])
    assert np.allclose(o["density"], direct, rtol=1e-12)
    assert np.all(o["density"] >= 0)
    assert o["mass"] == pytest.approx(1.0, abs=1e-3)
    assert o["is_density"] is True
    assert o["h"] == 0.35
    with pytest.raises(ValueError, match="positive"):
        wasserman_kde(g, x, h=0)


def test_kde_tracks_a_known_density_and_reports_all_three_rules():
    x = np.random.default_rng(11).normal(size=5000)
    g = np.linspace(-3, 3, 200)
    o = wasserman_kde(g, x)
    truth = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    err = np.max(np.abs(o["density"] - truth))
    # the residual sits at the mode, where any second-order kernel
    # estimate is biased downward by h^2 f''(x) mu_2 / 2; what matters
    # is that the chosen rule beats a deliberately mis-set one in
    # BOTH directions, which is the property (3.31) is claimed to have
    assert err < 0.04
    over = np.max(np.abs(wasserman_kde(g, x, h=5 * o["h"])["density"] - truth))
    under = np.max(np.abs(wasserman_kde(g, x, h=o["h"] / 20)["density"]
                          - truth))
    assert err < over
    assert err < under
    assert o["rule"] == "3.31"
    assert o["h"] < o["h_normal_reference"]
    assert o["h_iqr"] > 0


# ------------------------------------------------- MacKay, importance sampling


def test_importance_sampling_recovers_a_known_expectation():
    """E[X^2] = 1 under a standard normal target, estimated from
    Cauchy draws. The Cauchy is the heavy-tailed sampler MacKay
    recommends, and it should give a usable effective sample size."""
    xs = stats.cauchy.rvs(size=200_000, random_state=1)
    o = wasserman_importance_sampling(
        lambda x: x ** 2, stats.norm.pdf, stats.cauchy.pdf, samples=xs)
    assert o["estimate"] == pytest.approx(1.0, rel=0.05)
    assert o["self_normalised"] is True
    assert o["ess_fraction"] > 0.3


def test_the_self_normalised_estimator_needs_no_normalising_constants():
    """(29.22) divides by sum(w), so scaling P* or Q* by any constant
    leaves the estimate unchanged. That is the point of the
    construction, and it is exactly what the unnormalised
    alternative cannot do."""
    xs = stats.norm.rvs(size=20_000, random_state=2, scale=2.0)
    kw = dict(samples=xs)
    base = wasserman_importance_sampling(
        lambda x: x ** 2, stats.norm.pdf,
        lambda x: stats.norm.pdf(x, scale=2.0), **kw)["estimate"]
    scaled = wasserman_importance_sampling(
        lambda x: x ** 2, lambda x: 137.0 * stats.norm.pdf(x),
        lambda x: 0.004 * stats.norm.pdf(x, scale=2.0), **kw)["estimate"]
    assert scaled == pytest.approx(base, rel=1e-12)
    # the unnormalised estimator is NOT invariant to that rescaling
    un = wasserman_importance_sampling(
        lambda x: x ** 2, lambda x: 137.0 * stats.norm.pdf(x),
        lambda x: 0.004 * stats.norm.pdf(x, scale=2.0),
        normalised=True, **kw)["estimate"]
    assert not un == pytest.approx(base, rel=0.5)


def test_a_light_tailed_sampler_degrades_the_effective_sample_size():
    """MacKay's cautionary illustration: an importance sampler should
    have heavy tails. Sampling a heavy-tailed target from a
    light-tailed proposal concentrates the weight on a handful of
    draws, and the effective sample size is what shows it."""
    xs = stats.norm.rvs(size=50_000, random_state=4)
    heavy = wasserman_importance_sampling(
        lambda x: np.abs(x), lambda x: stats.t.pdf(x, df=1.5),
        stats.norm.pdf, samples=xs)
    light = wasserman_importance_sampling(
        lambda x: np.abs(x), stats.norm.pdf,
        stats.norm.pdf, samples=xs)
    assert heavy["ess_fraction"] < light["ess_fraction"]
    assert heavy["max_weight_share"] > light["max_weight_share"]
    assert "HEAVY TAILS" in heavy["heavy_tail_advice"]


def test_importance_sampling_refuses_an_invalid_sampler():
    xs = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="zero or negative"):
        wasserman_importance_sampling(lambda x: x, lambda x: np.ones_like(x),
                                      lambda x: np.zeros_like(x), samples=xs)
    with pytest.raises(ValueError, match="rvs method"):
        wasserman_importance_sampling(lambda x: x, lambda x: x, lambda x: x)


# ------------------------------------------------- ESL, bootstrap and bagging


def test_bootstrap_variance_divides_by_B_minus_one():
    """ESL (7.53) prints B - 1, for the same reason a sample variance
    carries n - 1: the replicates are centred at their own mean."""
    x = np.random.default_rng(13).normal(loc=3, scale=2, size=300)
    o = wasserman_bootstrap(x, np.mean, B=200, seed=1)
    assert o["variance_ddof0"] / o["variance_ddof1"] == pytest.approx(
        199 / 200, rel=1e-12)
    assert o["value"] == o["variance_ddof1"]
    assert wasserman_bootstrap(x, np.mean, B=200, seed=1,
                               ddof=0)["value"] == o["variance_ddof0"]
    with pytest.raises(ValueError, match="ddof"):
        wasserman_bootstrap(x, np.mean, B=50, ddof=2)


def test_bootstrap_variance_of_the_mean_matches_the_closed_form():
    """Var(Xbar) = sigma^2/n, which the bootstrap must recover -- this
    is the one case where the answer is known in advance."""
    rng = np.random.default_rng(17)
    n, sigma = 400, 2.0
    got = []
    for _ in range(20):
        x = rng.normal(scale=sigma, size=n)
        got.append(wasserman_bootstrap(x, np.mean, B=400,
                                       seed=int(rng.integers(1e6)))["value"])
    assert np.mean(got) == pytest.approx(sigma ** 2 / n, rel=0.1)


def test_plug_in_estimator_and_its_bootstrap_standard_error():
    """T(F_n) for the mean is Xbar and its standard error is
    sigma/sqrt(n); the bootstrap has to find that on its own."""
    x = np.random.default_rng(19).normal(loc=5, scale=3, size=500)
    o = wasserman_plug_in_estimator(x, np.mean, B=600)
    assert o["estimate"] == pytest.approx(float(np.mean(x)), rel=1e-12)
    assert o["se"] == pytest.approx(3 / np.sqrt(500), rel=0.15)
    lo, hi = o["ci_percentile"]
    assert lo < 5 < hi
    assert abs(o["bootstrap_bias"]) < 0.05
    assert "Hadamard" in o["validity_condition"]
    assert wasserman_plug_in_estimator(x, np.mean, se=False)["se"] is None


def test_plug_in_works_for_a_functional_that_is_not_the_mean():
    """Nothing in the plug-in principle is special to the mean. The
    median's asymptotic standard error is 1/(2 f(m) sqrt(n)), which
    for a standard normal is sqrt(pi/2)/sqrt(n)."""
    x = np.random.default_rng(23).normal(size=2000)
    o = wasserman_plug_in_estimator(x, np.median, B=500)
    assert o["estimate"] == pytest.approx(float(np.median(x)), rel=1e-12)
    assert o["se"] == pytest.approx(np.sqrt(np.pi / 2) / np.sqrt(2000),
                                    rel=0.2)


def test_bagging_does_nothing_for_a_linear_procedure():
    """ESL Sec. 8.7's sharp corollary. The replicates are identically
    distributed, so bagging can only move variance; and for a fit
    that is linear in y the bootstrap average converges back to the
    fit on the original data, so there is nothing left to move."""
    rng = np.random.default_rng(29)
    X = rng.normal(size=(150, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=0.5, size=150)
    o = wasserman_bagging(X, y, B=500)
    assert o["max_shift_from_single_fit"] < 0.05 * np.std(y)
    assert o["bagged_spread"] == pytest.approx(
        o["replicate_spread"] / 500, rel=1e-12)


def test_bagging_moves_a_deep_tree_a_great_deal():
    """The contrast that makes the previous test meaningful. A deep
    regression tree is high-variance and wildly nonlinear in y, and
    bagging shifts it by orders of magnitude more than it shifts a
    linear fit."""
    from morie.fn.cart import _build_tree, _predict_one

    rng = np.random.default_rng(31)
    X = rng.normal(size=(150, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=0.5, size=150)

    def tree(Xt, yt):
        t = _build_tree(Xt, yt, 0, 12, 2)
        return lambda Xn: np.array([_predict_one(t, r) for r in Xn])

    linear = wasserman_bagging(X, y, B=60, seed=2)
    nonlinear = wasserman_bagging(X, y, model=tree, B=60, seed=2)
    assert (nonlinear["max_shift_from_single_fit"]
            > 10 * linear["max_shift_from_single_fit"])
    assert nonlinear["replicate_spread"] > linear["replicate_spread"]
    assert nonlinear["n_oob_missing"] == 0


# ------------------------------------------------- M-estimation


def test_mle_recovers_normal_parameters_with_textbook_standard_errors():
    """The one model where every answer is known: for a normal
    sample the MLE of the mean has standard error sigma/sqrt(n) and
    the MLE of the standard deviation has sigma/sqrt(2n)."""
    n, mu, sigma = 800, 2.5, 1.5
    x = np.random.default_rng(37).normal(loc=mu, scale=sigma, size=n)
    o = wasserman_mle(x, lambda d, t: stats.norm.pdf(d, t[0], abs(t[1])),
                      [0.0, 1.0])
    assert o["estimate"][0] == pytest.approx(mu, abs=0.15)
    assert abs(o["estimate"][1]) == pytest.approx(sigma, abs=0.15)
    assert o["is_maximum"] is True
    assert o["se"][0] == pytest.approx(sigma / np.sqrt(n), rel=0.1)
    assert o["se"][1] == pytest.approx(sigma / np.sqrt(2 * n), rel=0.15)


def test_mle_reports_no_standard_error_when_it_did_not_find_a_maximum():
    """A non-positive-definite observed information means the point
    is not a maximum, and the standard error would be the square root
    of a negative number. Refusing to report one is the only honest
    option."""
    x = np.random.default_rng(41).normal(size=200)
    o = wasserman_mle(x, lambda d, t: stats.norm.pdf(d, t[0], 1.0), [0.0])
    assert o["is_maximum"] is True and o["se"] is not None
    # a likelihood flat in the parameter has a singular Hessian
    flat = wasserman_mle(x, lambda d, t: stats.norm.pdf(d, 0.0, 1.0)
                         + 0 * t[0], [0.5])
    assert flat["is_maximum"] is False
    assert flat["se"] is None
    assert "not a maximum" in flat["not_a_maximum_note"]


def test_mle_refuses_a_starting_value_where_the_likelihood_is_undefined():
    x = np.random.default_rng(43).normal(size=100)
    with pytest.raises(ValueError, match="not finite at theta0"):
        wasserman_mle(x, lambda d, t: np.full(d.shape, -1.0), [0.0])


# ------------------------------------------------- admissibility


def test_admissibility_is_dominance_and_needs_the_whole_table():
    """A rule is inadmissible when another is never worse and
    somewhere strictly better. Rule C below is beaten by both A and
    B; A and B each win at one state and so both survive."""
    o = wasserman_admissible([[1, 5], [2, 2], [3, 6]], names=["A", "B", "C"])
    assert list(o["admissible"]) == [True, True, False]
    assert set(o["dominated_by"]["C"]) == {"A", "B"}
    assert o["admissible_names"] == ["A", "B"]
    assert o["bool"] is False
    assert o["is_complete_class"] is False
    # minimax minimises the WORST-case risk, which is a different
    # criterion and picks B here (worst 2) over A (worst 5)
    assert o["minimax_rule"] == "B"
    assert o["minimax_risk"] == 2


def test_identical_rules_do_not_dominate_each_other():
    """The definition needs STRICT improvement somewhere. Two rules
    with the same risk everywhere are tied, and a tie is not
    dominance -- so both remain admissible."""
    o = wasserman_admissible([[1, 2], [1, 2]])
    assert o["bool"] is True
    assert list(o["admissible"]) == [True, True]
    assert o["dominated_by"] == {}


def test_a_constant_rule_can_be_admissible_without_being_good():
    """Admissibility is not optimality. A rule that is superb at one
    state and dreadful everywhere else is admissible as long as
    nothing beats it at that state, which is why admissibility alone
    is a weak recommendation."""
    o = wasserman_admissible([[0.0, 99.0], [1.0, 1.0]],
                             names=["silly", "sensible"])
    assert list(o["admissible"]) == [True, True]
    assert o["minimax_rule"] == "sensible"


def test_admissibility_validates_its_table():
    with pytest.raises(ValueError, match="finite"):
        wasserman_admissible([[1.0, np.nan], [2.0, 2.0]])
    with pytest.raises(ValueError, match="names has"):
        wasserman_admissible([[1, 2], [2, 1]], names=["only-one"])
    o = wasserman_admissible([[1, 2]])
    assert o["n_rules"] == 1 and o["bool"] is True
