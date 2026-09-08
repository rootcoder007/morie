"""Gibbons and Chakraborti nonparametric shelf.

Anchored on identities the estimators must satisfy exactly, on designs
with a known answer, and on the book's own statements. Page and equation
numbers are from the 5th edition.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1241r import gibbons_concordance_rho_link
from morie.fn.gb2312 import gibbons_edf_consistent
from morie.fn.gb_hg2 import gibbons_hodges_lehmann_2
from morie.fn.gb_hgl import gibbons_hodges_lehmann
from morie.fn.gb_wsp import gibbons_concordance_preference


# --------------------------------------------------------------------
# Hodges-Lehmann, one sample
# --------------------------------------------------------------------

def test_walsh_average_count_is_n_times_n_plus_one_over_two():
    for n in (1, 4, 6, 9):
        out = gibbons_hodges_lehmann(list(range(1, n + 1)))
        assert out["n_walsh"] == n * (n + 1) // 2


def test_walsh_averages_include_the_observations_themselves():
    # the i == k terms are the data; dropping them is a different
    # estimator, so they must be present
    out = gibbons_hodges_lehmann([2.0, 8.0])
    assert sorted(float(w) for w in out["walsh_averages"]) == [2.0, 5.0, 8.0]


def test_hodges_lehmann_is_exact_on_a_hand_computable_case():
    # data 1, 2, 5, 6, 9, 13 -> 21 Walsh averages; median is 5.5
    out = gibbons_hodges_lehmann([1, 2, 5, 6, 9, 13])
    assert out["estimate"] == pytest.approx(5.5)
    assert out["n_walsh"] == 21


def test_hodges_lehmann_is_translation_equivariant():
    x = [3.2, -1.0, 4.4, 9.1, 0.5, 2.2, 7.7]
    a = gibbons_hodges_lehmann(x)["estimate"]
    b = gibbons_hodges_lehmann([v + 10.0 for v in x])["estimate"]
    assert b == pytest.approx(a + 10.0)


def test_hodges_lehmann_is_scale_equivariant():
    x = [3.2, -1.0, 4.4, 9.1, 0.5, 2.2, 7.7]
    a = gibbons_hodges_lehmann(x)["estimate"]
    b = gibbons_hodges_lehmann([v * 3.0 for v in x])["estimate"]
    assert b == pytest.approx(a * 3.0)


def test_hodges_lehmann_resists_an_outlier_the_mean_cannot():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    clean = gibbons_hodges_lehmann(x)
    dirty = gibbons_hodges_lehmann(x + [1000.0])
    assert abs(dirty["estimate"] - clean["estimate"]) < 1.0
    assert dirty["mean"] > 100.0


def test_interval_coverage_is_exact_and_conservative():
    # the Walsh averages are discrete, so attained coverage exceeds the
    # nominal level rather than matching it
    out = gibbons_hodges_lehmann([1, 2, 5, 6, 9, 13], alpha=0.05)
    assert out["coverage"] >= 0.95
    lo, hi = out["ci"]
    assert lo <= out["estimate"] <= hi


def test_interval_actually_covers_at_its_stated_rate():
    # symmetric design with a known median of 0
    hits = 0
    reps = 300
    for s in range(reps):
        rng = np.random.default_rng(s)
        x = rng.normal(size=12)
        lo, hi = gibbons_hodges_lehmann(x)["ci"]
        hits += lo <= 0.0 <= hi
    assert hits / reps >= 0.94


def test_hodges_lehmann_beats_the_mean_under_heavy_tails():
    # Cauchy: the mean has no variance, the HL estimator does
    hl, mn = [], []
    for s in range(200):
        rng = np.random.default_rng(s)
        x = rng.standard_cauchy(size=15)
        hl.append(gibbons_hodges_lehmann(x)["estimate"])
        mn.append(float(np.mean(x)))
    assert np.std(hl) < np.std(mn) / 5


# --------------------------------------------------------------------
# Hodges-Lehmann, two samples
# --------------------------------------------------------------------

def test_two_sample_estimator_is_the_median_of_all_differences():
    x, y = [1, 6, 7], [2, 4, 9, 10, 12]
    out = gibbons_hodges_lehmann_2(x, y)
    diffs = sorted(b - a for b in y for a in x)
    assert out["n_differences"] == len(x) * len(y)
    assert out["estimate"] == pytest.approx(float(np.median(diffs)))
    assert out["estimate"] == pytest.approx(3.0)


def test_two_sample_estimator_recovers_a_known_shift():
    ests = []
    for s in range(60):
        rng = np.random.default_rng(s)
        x = rng.normal(size=25)
        y = rng.normal(size=25) + 2.5
        ests.append(gibbons_hodges_lehmann_2(x, y)["estimate"])
    assert abs(float(np.mean(ests)) - 2.5) < 0.15


def test_shift_estimate_flips_sign_when_the_samples_swap():
    x = [1.0, 4.0, 6.0, 9.0]
    y = [3.0, 7.0, 8.0, 12.0]
    a = gibbons_hodges_lehmann_2(x, y)["estimate"]
    b = gibbons_hodges_lehmann_2(y, x)["estimate"]
    assert b == pytest.approx(-a)


def test_a_common_shift_moves_the_estimate_by_exactly_that_amount():
    x = [1.0, 4.0, 6.0, 9.0, 11.0]
    y = [3.0, 7.0, 8.0, 12.0, 14.0]
    a = gibbons_hodges_lehmann_2(x, y)["estimate"]
    b = gibbons_hodges_lehmann_2(x, [v + 5.0 for v in y])["estimate"]
    assert b == pytest.approx(a + 5.0)


def test_unequal_spread_is_flagged_because_no_single_shift_exists():
    rng = np.random.default_rng(0)
    same = gibbons_hodges_lehmann_2(rng.normal(size=40),
                                    rng.normal(size=40) + 1.0)
    diff = gibbons_hodges_lehmann_2(rng.normal(size=40),
                                    rng.normal(size=40) * 8.0 + 1.0)
    assert same["shift_plausible"] is True
    assert diff["shift_plausible"] is False
    assert "translation" in diff["shift_note"]


def test_two_sample_interval_covers_the_true_shift():
    hits = 0
    reps = 200
    for s in range(reps):
        rng = np.random.default_rng(s)
        x = rng.normal(size=12)
        y = rng.normal(size=12) + 1.5
        lo, hi = gibbons_hodges_lehmann_2(x, y)["ci"]
        hits += lo <= 1.5 <= hi
    assert hits / reps >= 0.94


# --------------------------------------------------------------------
# W and the average rank correlation
# --------------------------------------------------------------------

def test_perfect_concordance_maps_to_rho_one():
    assert gibbons_concordance_rho_link(1.0, 4)["rho_av"] == pytest.approx(1.0)


def test_zero_concordance_maps_to_the_floor():
    # equation (12.4.6) at W = 0 gives the smallest attainable r_av
    for k in (2, 3, 5, 10):
        out = gibbons_concordance_rho_link(0.0, k)
        assert out["rho_av"] == pytest.approx(-1.0 / (k - 1))
        assert out["rho_min"] == pytest.approx(-1.0 / (k - 1))


def test_the_two_forms_are_inverses():
    # (12.4.6) and (12.4.7) must round-trip exactly
    for k in (2, 3, 7):
        for W in (0.0, 0.25, 0.5, 0.9, 1.0):
            out = gibbons_concordance_rho_link(W, k)
            assert out["inverse_check"] == pytest.approx(W)


def test_rho_minus_one_is_unattainable_beyond_two_rankings():
    # three rankings cannot all disagree perfectly with one another
    assert gibbons_concordance_rho_link(0.0, 2)["rho_min"] == pytest.approx(-1.0)
    assert gibbons_concordance_rho_link(0.0, 3)["rho_min"] > -1.0


def test_null_moments_match_equation_12_4_8():
    out = gibbons_concordance_rho_link(0.4, k=5, n=9)
    assert out["expected_W"] == pytest.approx(1 / 5)
    assert out["var_W"] == pytest.approx(2 * (5 - 1) / (5 ** 3 * (9 - 1)))
    assert out["chi2"] == pytest.approx(5 * 8 * 0.4)
    assert out["df"] == 8


def test_rho_link_validation():
    with pytest.raises(ValueError, match="at least 2 rankings"):
        gibbons_concordance_rho_link(0.5, 1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gibbons_concordance_rho_link(1.5, 3)


# --------------------------------------------------------------------
# consensus ordering
# --------------------------------------------------------------------

def test_unanimous_rankings_reproduce_themselves():
    r = [[1, 2, 3, 4]] * 5
    out = gibbons_concordance_preference(r)
    assert list(out["preference_rank"]) == [1.0, 2.0, 3.0, 4.0]
    assert out["W"] == pytest.approx(1.0)
    assert out["rho_bar"] == pytest.approx(1.0)


def test_consensus_follows_the_rank_totals():
    # object 2 has the smallest total, so it must come first
    r = [[3, 1, 2], [3, 1, 2], [2, 1, 3]]
    out = gibbons_concordance_preference(r)
    assert int(out["order"][0]) == 1
    assert out["rank_sums"][1] == pytest.approx(3.0)


def test_the_consensus_maximises_average_agreement():
    # the book's optimality claim, checked against every permutation
    from itertools import permutations
    rng = np.random.default_rng(3)
    R = np.vstack([rng.permutation(5) + 1 for _ in range(4)])
    out = gibbons_concordance_preference(R)
    best = float(out["rho_bar"])

    def avg_rho(cand):
        cand = np.asarray(cand, dtype=float)
        vals = []
        for row in R:
            a = row - row.mean()
            b = cand - cand.mean()
            vals.append(float(a @ b) / np.sqrt(float(a @ a) * float(b @ b)))
        return float(np.mean(vals))

    for perm in permutations(range(1, 6)):
        assert avg_rho(perm) <= best + 1e-9


def test_expected_W_under_independence_is_one_over_k():
    r = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    out = gibbons_concordance_preference(r)
    assert out["expected_W_under_independence"] == pytest.approx(1 / 3)
    # these three rankings are mutually maximally disagreeing
    assert out["W"] == pytest.approx(0.0)


def test_W_matches_the_link_to_pairwise_rho():
    rng = np.random.default_rng(7)
    R = np.vstack([rng.permutation(6) + 1 for _ in range(5)])
    out = gibbons_concordance_preference(R)
    link = gibbons_concordance_rho_link(out["W"], 5)
    # equation (12.4.6) must agree with the pairwise average computed
    # directly, which is an identity rather than an approximation
    assert link["rho_av"] == pytest.approx(out["rho_av_pairwise"], abs=1e-9)


def test_a_dissenting_observer_is_visible():
    r = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]
    out = gibbons_concordance_preference(r)
    rho = out["rho_by_observer"]
    assert rho[3] < 0 < rho[0]


def test_preference_validation():
    with pytest.raises(ValueError, match="at least 2 rankings"):
        gibbons_concordance_preference([[1, 2, 3]])
    with pytest.raises(ValueError, match="at least 2 objects"):
        gibbons_concordance_preference([[1], [1]])


# --------------------------------------------------------------------
# empirical distribution function
# --------------------------------------------------------------------

def test_edf_is_a_step_function_reaching_one():
    out = gibbons_edf_consistent([3.0, 1.0, 2.0])
    assert list(np.round(out["edf"], 10)) == [pytest.approx(1 / 3),
                                              pytest.approx(2 / 3),
                                              pytest.approx(1.0)]
    assert out["jump"] == pytest.approx(1 / 3)


def test_edf_is_unbiased_at_every_point():
    # E[S_n(x)] = F(x) exactly; check against the uniform, where F(x) = x
    grid = np.array([0.2, 0.5, 0.8])
    acc = np.zeros(3)
    reps = 800
    for s in range(reps):
        rng = np.random.default_rng(s)
        acc += gibbons_edf_consistent(rng.uniform(size=40), at=grid)["edf"]
    assert np.max(np.abs(acc / reps - grid)) < 0.02


def test_variance_is_f_times_one_minus_f_over_n():
    out = gibbons_edf_consistent(np.linspace(0, 1, 101), at=[0.5])
    f = float(out["edf"][0])
    assert out["variance"][0] == pytest.approx(f * (1 - f) / 101)


def test_variance_peaks_at_the_median_and_vanishes_in_the_tails():
    rng = np.random.default_rng(0)
    out = gibbons_edf_consistent(rng.normal(size=400),
                                 at=[-3.0, 0.0, 3.0])
    v = out["variance"]
    assert v[1] > v[0] and v[1] > v[2]


def test_consistency_shows_up_as_a_shrinking_sup_deviation():
    sups = []
    for n in (50, 500, 5000):
        rng = np.random.default_rng(1)
        x = rng.uniform(size=n)
        grid = np.linspace(0.01, 0.99, 199)
        e = gibbons_edf_consistent(x, at=grid)["edf"]
        sups.append(float(np.max(np.abs(e - grid))))
    assert sups[0] > sups[1] > sups[2]


def test_the_dkw_band_covers_the_truth_simultaneously():
    hits = 0
    reps = 200
    grid = np.linspace(0.01, 0.99, 99)
    for s in range(reps):
        rng = np.random.default_rng(s)
        out = gibbons_edf_consistent(rng.uniform(size=60), at=grid, alpha=0.05)
        inside = np.all((out["band_lower"] <= grid)
                        & (grid <= out["band_upper"]))
        hits += bool(inside)
    # simultaneous over all x, so coverage is conservative by design
    assert hits / reps >= 0.95


def test_dkw_epsilon_matches_massarts_constant():
    out = gibbons_edf_consistent(np.arange(100.0), alpha=0.05)
    import math
    assert out["dkw_epsilon"] == pytest.approx(
        math.sqrt(math.log(2 / 0.05) / 200.0)
    )


def test_edf_validation():
    with pytest.raises(ValueError, match="at least 1 observation"):
        gibbons_edf_consistent([])
    with pytest.raises(ValueError, match="non-finite"):
        gibbons_edf_consistent([1.0, np.nan])
