"""Median voter, dual-frame totals, and the median lethal dose.

Sources: Black (1948) *JPE* 56(1):23-34; Hartley (1962) *Proc Soc Stat
Sect ASA* 203-206 and Lohr and Rao (2000) *JASA* 95:271-280; Finney
(1971) *Probit Analysis* 3rd ed. Ch 3-4 and Fieller (1954) *JRSS B*
16(2):175-185.
"""

import math

import numpy as np
import pytest

from morie.fn.ld50r import acute_toxicity_ld50, effective_dose
from morie.fn.mdvtr import condorcet_winner, median_voter
from morie.fn.smplep import optimal_overlap_weight, sample_overlap


# --------------------------------------------------------------------
# Median voter
# --------------------------------------------------------------------

def test_the_winner_is_the_median_not_the_mean():
    # one extremist moves the mean and leaves the winner alone; this is
    # the entire content of the theorem
    out = median_voter([1.0, 2.0, 3.0, 4.0, 100.0])
    assert out["estimate"] == 3.0
    assert out["mean"] == pytest.approx(22.0)


def test_moving_an_extremist_further_out_changes_nothing():
    a = median_voter([1.0, 2.0, 3.0, 4.0, 100.0])["estimate"]
    b = median_voter([1.0, 2.0, 3.0, 4.0, 1e9])["estimate"]
    assert a == b == 3.0


def test_the_median_beats_every_alternative_pairwise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=101)
    out = median_voter(x, alternatives=[-2.0, -0.5, 0.5, 2.0])
    assert out["condorcet_verified"] is True
    assert out["alternatives_beaten"].all()


def test_a_platform_nearer_the_median_beats_one_further_away():
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    med = median_voter(x)["estimate"]
    near, far = med + 0.1, med + 5.0
    for_near = int(np.sum(np.abs(x - near) < np.abs(x - far)))
    assert for_near > len(x) / 2


def test_an_even_electorate_gives_an_interval_of_winners():
    out = median_voter([1.0, 2.0, 3.0, 4.0])
    assert out["unique_winner"] is False
    assert out["median_interval"] == (2.0, 3.0)
    assert any("Condorcet winner" in w for w in out.warnings)


def test_an_odd_electorate_gives_a_point():
    out = median_voter([1.0, 2.0, 3.0])
    assert out["unique_winner"] is True
    assert out["median_interval"] == (2.0, 2.0)
    assert not any("even electorate" in w for w in out.warnings)


def test_the_density_standard_error_matches_theory_for_a_normal():
    # for N(0,1) the asymptotic SE of the median is sqrt(pi/2)/sqrt(n),
    # so the density-based and normal-based forms should agree here
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    out = median_voter(x)
    theory = math.sqrt(math.pi / 2) / math.sqrt(4000)
    assert out["se"] == pytest.approx(theory, rel=0.12)
    assert out["se_normal"] == pytest.approx(theory, rel=0.06)


def test_the_normal_formula_badly_overstates_for_a_heavy_tail():
    # the median only responds to density at the centre; the 1.2533
    # formula reads the tails as spread and inflates the SE
    rng = np.random.default_rng(2)
    x = rng.standard_t(2.0, size=4000)
    out = median_voter(x)
    assert out["se_normal"] / out["se"] > 1.3
    assert any("far from normal" in w for w in out.warnings)


def test_the_density_error_tracks_the_sampling_variability():
    rng = np.random.default_rng(3)
    meds = [float(np.median(rng.standard_t(2.0, size=2000)))
            for _ in range(400)]
    emp = float(np.std(meds, ddof=1))
    rep = median_voter(rng.standard_t(2.0, size=2000))["se"]
    assert rep == pytest.approx(emp, rel=0.25)


def test_no_spurious_normality_warning_for_normal_data():
    rng = np.random.default_rng(4)
    out = median_voter(rng.normal(size=3000))
    assert not any("far from normal" in w for w in out.warnings)


def test_the_order_statistic_interval_covers_without_any_assumption():
    rng = np.random.default_rng(5)
    hits = 0
    reps = 800
    for _ in range(reps):
        x = rng.standard_cauchy(size=51)   # no mean, no variance
        o = median_voter(x)
        hits += o["ci_exact_lower"] <= 0.0 <= o["ci_exact_upper"]
    assert hits / reps > 0.95


def test_the_exact_interval_reports_its_achieved_level():
    out = median_voter(np.arange(51.0))
    assert 0.95 <= out["exact_coverage"] < 1.0
    assert out["ci_exact_lower"] < out["estimate"] < out["ci_exact_upper"]


def test_the_alternatives_check_cannot_fail_and_says_so():
    # preferences here are built from distance on a line, and Euclidean
    # preferences in one dimension are single-peaked by construction, so
    # the median wins every contest as arithmetic. The check verifies
    # the implementation, not the assumption, and the payload admits it.
    rng = np.random.default_rng(20)
    for s in range(30):
        x = rng.normal(size=41) * (s + 1)
        out = median_voter(x, alternatives=rng.normal(size=6) * 10)
        assert out["condorcet_verified"] is True
        assert out["check_is_definitional"] is True


def test_a_genuine_condorcet_cycle_has_no_winner():
    # the classic paradox: A>B>C, B>C>A, C>A>B. Every option loses to
    # some other and the majority relation runs in a circle.
    U = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
    out = condorcet_winner(U)
    assert out["exists"] is False
    assert out["cyclic"] is True
    assert out["winner"] is None
    assert list(out["net_wins"]) == [1, 1, 1]
    assert any("No Condorcet winner" in w for w in out.warnings)


def test_a_clear_majority_favourite_is_found():
    U = np.array([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    out = condorcet_winner(U, platforms=["left", "centre", "right"])
    assert out["exists"] is True
    assert out["winner"] == "left"


def test_single_peaked_utilities_always_produce_a_winner():
    # build utilities as negative distance -- single-peaked by
    # construction -- and confirm the winner is the median platform
    rng = np.random.default_rng(21)
    plats = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ideal = rng.normal(size=101)
    U = -np.abs(ideal[:, None] - plats[None, :])
    out = condorcet_winner(U, platforms=plats)
    assert out["exists"] is True
    assert abs(out["winner"] - np.median(ideal)) <= 1.0


def test_condorcet_input_validation():
    with pytest.raises(ValueError, match="at least two options"):
        condorcet_winner(np.ones((5, 1)))


def test_median_voter_edge_cases():
    assert median_voter([])["n"] == 0
    one = median_voter([2.5])
    assert one["estimate"] == 2.5
    assert math.isnan(one["se"])
    with pytest.raises(ValueError, match="alpha"):
        median_voter([1.0, 2.0, 3.0], alpha=0.0)


# --------------------------------------------------------------------
# Dual-frame estimation
# --------------------------------------------------------------------

def test_the_naive_pool_double_counts_the_overlap():
    ya = [1.0, 1.0, 1.0, 1.0]
    yb = [1.0, 1.0, 1.0]
    out = sample_overlap(ya, yb, [0, 0, 1, 1], [1, 1, 0], theta=0.5)
    assert out["estimate"] == 5.0
    assert out["naive_pooled_total"] == 7.0
    assert out["overlap_double_count"] == 2.0


def test_the_estimator_is_unbiased_for_every_theta():
    # unbiasedness does not depend on theta at all; only precision does
    ya = [2.0, 2.0, 2.0, 2.0]
    yb = [2.0, 2.0, 2.0]
    ests = [sample_overlap(ya, yb, [0, 0, 1, 1], [1, 1, 0],
                           theta=t)["estimate"]
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(e == pytest.approx(ests[0]) for e in ests)


def test_theta_zero_and_one_take_the_overlap_from_one_frame_only():
    ya = [1.0, 5.0]
    yb = [9.0, 1.0]
    a = sample_overlap(ya, yb, [0, 1], [1, 0], theta=1.0)
    b = sample_overlap(ya, yb, [0, 1], [1, 0], theta=0.0)
    assert a["overlap_estimate"] == 5.0
    assert b["overlap_estimate"] == 9.0


def test_the_optimal_weight_favours_the_more_precise_frame():
    # theta multiplies frame A, so it rises as frame B gets noisier
    assert optimal_overlap_weight(1.0, 9.0) == pytest.approx(0.9)
    assert optimal_overlap_weight(9.0, 1.0) == pytest.approx(0.1)
    assert optimal_overlap_weight(2.0, 2.0) == pytest.approx(0.5)
    assert optimal_overlap_weight(0.0, 0.0) == pytest.approx(0.5)


def test_the_optimal_weight_really_minimises_the_variance():
    rng = np.random.default_rng(10)
    ya = rng.normal(10.0, 1.0, size=40)
    yb = rng.normal(10.0, 4.0, size=40)
    da = np.r_[np.zeros(20), np.ones(20)]
    db = np.r_[np.ones(20), np.zeros(20)]
    opt = sample_overlap(ya, yb, da, db)
    for t in np.linspace(0, 1, 21):
        v = sample_overlap(ya, yb, da, db, theta=float(t))["variance"]
        assert v >= opt["variance"] - 1e-9


def test_a_suboptimal_theta_is_flagged_but_still_unbiased():
    rng = np.random.default_rng(11)
    ya = rng.normal(10.0, 1.0, size=40)
    yb = rng.normal(10.0, 6.0, size=40)
    da = np.r_[np.zeros(20), np.ones(20)]
    db = np.r_[np.ones(20), np.zeros(20)]
    bad = sample_overlap(ya, yb, da, db, theta=0.0)
    assert bad["variance_ratio_vs_optimal"] > 1.05
    assert any("still unbiased" in w for w in bad.warnings)


def test_the_estimator_recovers_a_known_population_total():
    # frame A covers units 0..69, frame B covers 50..99, overlap 50..69
    rng = np.random.default_rng(12)
    pop = rng.normal(5.0, 1.0, size=100)
    truth = float(pop.sum())
    ests = []
    for r in range(300):
        g = np.random.default_rng(1000 + r)
        ia = g.choice(70, size=35, replace=False)
        ib = g.choice(np.arange(50, 100), size=25, replace=False)
        out = sample_overlap(pop[ia], pop[ib],
                             (ia >= 50).astype(float),
                             (ib < 70).astype(float),
                             weights_a=np.full(35, 70 / 35),
                             weights_b=np.full(25, 50 / 25),
                             theta=0.5)
        ests.append(out["estimate"])
    assert abs(float(np.mean(ests)) - truth) < 0.03 * truth


def test_the_naive_pool_is_biased_not_noisy():
    # the inflation is stable across replications, which is what makes
    # it invisible without the domain decomposition
    rng = np.random.default_rng(13)
    pop = rng.normal(5.0, 1.0, size=100)
    naive, correct = [], []
    for r in range(200):
        g = np.random.default_rng(2000 + r)
        ia = g.choice(70, size=35, replace=False)
        ib = g.choice(np.arange(50, 100), size=25, replace=False)
        o = sample_overlap(pop[ia], pop[ib], (ia >= 50).astype(float),
                           (ib < 70).astype(float),
                           weights_a=np.full(35, 2.0),
                           weights_b=np.full(25, 2.0), theta=0.5)
        naive.append(o["naive_pooled_total"])
        correct.append(o["estimate"])
    # the inflation is large relative to its own scatter, which is the
    # point: a bias this stable never looks like an error
    inflation = float(np.mean(naive)) - float(np.mean(correct))
    assert inflation > 0
    assert inflation > 4 * float(np.std(naive))
    assert float(np.std(naive)) / float(np.mean(naive)) < 0.1


def test_a_missing_overlap_is_reported():
    out = sample_overlap([1.0, 2.0], [3.0, 4.0], [0, 0], [0, 0])
    assert out["n_overlap_a"] == 0
    assert any("no overlap units" in w for w in out.warnings)


def test_design_weights_scale_the_total():
    a = sample_overlap([1.0, 1.0], [1.0, 1.0], [0, 1], [1, 0], theta=0.5)
    b = sample_overlap([1.0, 1.0], [1.0, 1.0], [0, 1], [1, 0], theta=0.5,
                       weights_a=[3.0, 3.0], weights_b=[3.0, 3.0])
    assert b["estimate"] == pytest.approx(3.0 * a["estimate"])


def test_dual_frame_input_validation():
    with pytest.raises(ValueError, match="overlap_a has length"):
        sample_overlap([1.0, 2.0], [3.0], [0], [1])
    with pytest.raises(ValueError, match="binary"):
        sample_overlap([1.0], [2.0], [2], [1])
    with pytest.raises(ValueError, match="theta must lie"):
        sample_overlap([1.0], [2.0], [0], [1], theta=1.5)
    with pytest.raises(ValueError, match="weights must match"):
        sample_overlap([1.0], [2.0], [0], [1], weights_a=[1.0, 1.0])
    with pytest.raises(ValueError, match="must be positive"):
        sample_overlap([1.0], [2.0], [0], [1], weights_a=[0.0])
    with pytest.raises(ValueError, match="at least one unit"):
        sample_overlap([], [], [], [])


# --------------------------------------------------------------------
# Median lethal dose
# --------------------------------------------------------------------

def quantal(ld50=4.0, slope=2.0, doses=(0.5, 1, 2, 4, 8, 16, 32),
            per_group=60, seed=0, link="probit"):
    """Simulate a quantal assay with a known LD50."""
    rng = np.random.default_rng(seed)
    d = np.asarray(doses, dtype=float)
    eta = slope * (np.log(d) - math.log(ld50))
    if link == "probit":
        p = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in eta])
    else:
        p = 1 / (1 + np.exp(-eta))
    k = rng.binomial(per_group, p)
    return d, k.astype(float), np.full(d.size, float(per_group))


def test_the_ld50_recovers_the_design_truth():
    d, k, n = quantal(ld50=4.0, per_group=400, seed=1)
    out = acute_toxicity_ld50(d, k, n)
    assert out["estimate"] == pytest.approx(4.0, rel=0.06)
    assert out["converged"] is True


def test_the_fitted_slope_recovers_the_design_slope():
    d, k, n = quantal(slope=2.0, per_group=400, seed=2)
    out = acute_toxicity_ld50(d, k, n)
    assert out["slope"] == pytest.approx(2.0, rel=0.15)


def test_the_fieller_interval_covers_at_the_nominal_rate():
    # 1200 replications: at 300 the Monte Carlo error on a coverage
    # estimate is ~1.3 points, enough to read a correct 95 as a 92
    hits = 0
    reps = 1200
    for s in range(reps):
        d, k, n = quantal(ld50=4.0, per_group=80, seed=100 + s)
        o = acute_toxicity_ld50(d, k, n)
        if o["bounded"]:
            hits += o["ci_lower"] <= 4.0 <= o["ci_upper"]
        else:
            hits += 1        # an unbounded interval trivially covers
    assert hits / reps > 0.93


def test_the_interval_is_unbounded_when_the_slope_is_not_significant():
    # a flat response carries no information about where the median is,
    # and Fieller says so instead of returning two tidy numbers
    d = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    k = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    n = np.full(5, 20.0)
    out = acute_toxicity_ld50(d, k, n)
    assert out["fieller_g"] >= 1.0
    assert out["bounded"] is False
    assert not math.isfinite(out["ci_lower"]) or math.isnan(out["ci_lower"])
    assert any("unbounded" in w for w in out.warnings)


def test_the_delta_method_stays_finite_where_fieller_goes_unbounded():
    # the point of preferring Fieller. With a weak but non-zero slope
    # the delta method returns a tidy finite standard error while
    # Fieller correctly refuses to bound the median.
    d = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    k = np.array([9.0, 10.0, 10.0, 11.0, 11.0])
    n = np.full(5, 20.0)
    out = acute_toxicity_ld50(d, k, n)
    assert out["slope"] != 0.0
    assert math.isfinite(out["se_log"])
    assert out["bounded"] is False
    assert out["fieller_g"] >= 1.0


def test_fieller_g_falls_as_the_slope_becomes_better_determined():
    small = acute_toxicity_ld50(*quantal(per_group=20, seed=3))
    large = acute_toxicity_ld50(*quantal(per_group=800, seed=3))
    assert large["fieller_g"] < small["fieller_g"]
    assert large["bounded"] is True


def test_the_interval_narrows_with_more_subjects():
    small = acute_toxicity_ld50(*quantal(per_group=40, seed=4))
    large = acute_toxicity_ld50(*quantal(per_group=1000, seed=4))
    assert (large["ci_upper"] - large["ci_lower"]) < \
           (small["ci_upper"] - small["ci_lower"])


def test_the_ld50_barely_depends_on_the_link():
    d, k, n = quantal(per_group=400, seed=5)
    pro = acute_toxicity_ld50(d, k, n, link="probit")
    log = acute_toxicity_ld50(d, k, n, link="logit")
    assert pro["estimate"] == pytest.approx(log["estimate"], rel=0.03)
    assert pro["link_sensitivity"] < 0.05


def test_the_tails_do_depend_on_the_link():
    # the LD50 is nearly link-free; the LD01 is not, which is why an
    # extreme quantile read off the wrong link is a real error
    d, k, n = quantal(per_group=400, seed=6)
    pro = acute_toxicity_ld50(d, k, n, link="probit", level=0.01)
    log = acute_toxicity_ld50(d, k, n, link="logit", level=0.01)
    mid_pro = acute_toxicity_ld50(d, k, n, link="probit")
    assert pro["link_sensitivity"] > mid_pro["link_sensitivity"] * 3


def test_a_doubled_potency_halves_the_ld50():
    a = acute_toxicity_ld50(*quantal(ld50=4.0, per_group=800, seed=7))
    b = acute_toxicity_ld50(*quantal(ld50=8.0, per_group=800, seed=7))
    assert b["estimate"] / a["estimate"] == pytest.approx(2.0, rel=0.1)


def test_heterogeneity_is_detected_and_reported():
    # responses driven by an unmodelled cage effect, not by dose alone
    d = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    k = np.array([2.0, 30.0, 5.0, 55.0, 20.0, 59.0])
    n = np.full(6, 60.0)
    out = acute_toxicity_ld50(d, k, n)
    assert out["heterogeneity_factor"] > 1.0
    assert out["heterogeneity_p"] < 0.05
    assert any("heterogeneity factor" in w for w in out.warnings)


def test_the_heterogeneity_warning_does_not_fire_on_correct_models():
    # a factor above 1 means nothing on its own -- the deviance has
    # expectation df, so the ratio exceeds 1 in about a fifth of
    # correctly-specified samples. The warning must key on the tail
    # probability, not the ratio, or it cries wolf 20 % of the time.
    fired = above_one = 0
    reps = 300
    for s in range(reps):
        out = acute_toxicity_ld50(*quantal(per_group=400, seed=9000 + s))
        fired += any("heterogeneity factor" in w for w in out.warnings)
        above_one += out["heterogeneity_factor"] > 1.0
    assert above_one / reps > 0.1        # the naive rule would misfire
    assert fired / reps < 0.09           # the tail test does not


def test_the_heterogeneity_test_is_uniform_when_every_group_informs():
    # narrow doses keep all fitted probabilities away from 0 and 1, so
    # every group contributes deviance and the chi-square reference is
    # the right one
    doses = (2.0, 2.8, 3.4, 4.0, 4.7, 5.7, 8.0)
    ps = np.array([acute_toxicity_ld50(
        *quantal(doses=doses, per_group=400, seed=9500 + s))[
            "heterogeneity_p"] for s in range(300)])
    assert 0.40 < float(np.mean(ps)) < 0.60
    assert 0.02 < float(np.mean(ps < 0.05)) < 0.10


def test_saturated_dose_groups_make_the_test_conservative():
    # a group fitted at essentially 0 or 1 contributes no deviance but
    # still spends a degree of freedom, so the null p-values pile up
    # high and the test under-warns. Safe direction, but not nominal.
    wide = np.array([acute_toxicity_ld50(
        *quantal(doses=(0.5, 1, 2, 4, 8, 16, 32), per_group=400,
                 seed=7000 + s))["heterogeneity_p"] for s in range(300)])
    narrow = np.array([acute_toxicity_ld50(
        *quantal(doses=(2.0, 2.8, 3.4, 4.0, 4.7, 5.7, 8.0),
                 per_group=400, seed=7000 + s))["heterogeneity_p"]
        for s in range(300)])
    assert float(np.mean(wide)) > float(np.mean(narrow)) + 0.1
    assert float(np.mean(wide < 0.05)) < float(np.mean(narrow < 0.05))


def test_saturated_groups_are_flagged():
    d = np.array([1.0, 4.0, 16.0])
    k = np.array([0.0, 10.0, 20.0])
    n = np.full(3, 20.0)
    out = acute_toxicity_ld50(d, k, n)
    assert any("0 or 100 per cent" in w for w in out.warnings)


def test_effective_dose_inverts_the_curve_exactly():
    # with a known intercept and slope the ED is arithmetic, not a fit
    ed = effective_dose(intercept=-2.0, slope=1.0, cov=np.eye(2) * 1e-8,
                        level=0.5, link="probit", log_scale=False)
    assert ed["ed"] == pytest.approx(2.0)
    assert ed["bounded"] is True


def test_effective_dose_at_other_levels_moves_the_right_way():
    cov = np.eye(2) * 1e-8
    lo = effective_dose(-2.0, 1.0, cov, level=0.1, link="probit",
                        log_scale=False)["ed"]
    mid = effective_dose(-2.0, 1.0, cov, level=0.5, link="probit",
                         log_scale=False)["ed"]
    hi = effective_dose(-2.0, 1.0, cov, level=0.9, link="probit",
                        log_scale=False)["ed"]
    assert lo < mid < hi


def test_ld50_input_validation():
    with pytest.raises(ValueError, match="must agree in length"):
        acute_toxicity_ld50([1.0, 2.0], [1.0], [10.0, 10.0])
    with pytest.raises(ValueError, match="at least two dose groups"):
        acute_toxicity_ld50([1.0], [1.0], [10.0])
    with pytest.raises(ValueError, match="n_dead must lie"):
        acute_toxicity_ld50([1.0, 2.0], [11.0, 1.0], [10.0, 10.0])
    with pytest.raises(ValueError, match="n_total must be positive"):
        acute_toxicity_ld50([1.0, 2.0], [0.0, 1.0], [0.0, 10.0])
    with pytest.raises(ValueError, match="dose must be positive"):
        acute_toxicity_ld50([0.0, 2.0], [1.0, 5.0], [10.0, 10.0])
    with pytest.raises(ValueError, match="link"):
        acute_toxicity_ld50([1.0, 2.0], [1.0, 5.0], [10.0, 10.0],
                            link="cloglog")
    with pytest.raises(ValueError, match="level must lie"):
        effective_dose(-2.0, 1.0, np.eye(2), level=1.0)
    with pytest.raises(ValueError, match="cov must be 2x2"):
        effective_dose(-2.0, 1.0, np.eye(3))
