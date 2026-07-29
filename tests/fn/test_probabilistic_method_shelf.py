"""The probabilistic method, with every bound checked against the truth.

These are all one-sided inequalities, so there are two things to test
and both matter: the bound must never be violated by a directly
computed probability, and it must not be vacuous. A bound of 1 holds
universally and says nothing.

Sources: Alon and Spencer (2016) *The Probabilistic Method* 4th ed.;
Erdos (1947); Erdos and Lovasz (1975); Chernoff (1952); Azuma (1967).
"""

import math
import random

import pytest

from morie.fn.prbmth import (
    alteration_ramsey,
    azuma_bound,
    chernoff_bound,
    first_moment_ramsey,
    lovasz_local_lemma,
    second_moment_threshold,
    union_bound_exists,
)
from morie.fn.ramthy import ramsey_lower_bound_probabilistic, ramsey_number


# --------------------------------------------------------------------
# First moment
# --------------------------------------------------------------------

def test_the_union_bound_triggers_exactly_at_expectation_one():
    assert union_bound_exists(10, 0.05)["exists"] is True
    assert union_bound_exists(100, 0.05)["exists"] is False
    assert union_bound_exists(20, 0.05)["exists"] is False   # exactly 1


def test_failing_the_union_bound_is_not_a_proof_of_nonexistence():
    out = union_bound_exists(100, 0.5)
    assert out["exists"] is False
    assert "does NOT show" in out["method"] or \
        "does NOT show" in out.interpretation


def test_the_first_moment_ramsey_bound_agrees_with_the_ramsey_shelf():
    for k in (3, 4, 6, 10, 15, 20):
        assert first_moment_ramsey(k)["bound"] == \
            ramsey_lower_bound_probabilistic(k)["bound"]


def test_the_bound_lies_below_the_true_value_where_known():
    for k in (3, 4):
        assert first_moment_ramsey(k)["bound"] < \
            ramsey_number(k, k)["value"]


def test_the_expected_count_at_the_bound_is_below_one():
    for k in (4, 6, 10, 20):
        assert first_moment_ramsey(k)["expected_at_bound"] < 1.0


def test_a_capped_search_says_so_rather_than_returning_the_ceiling():
    small = first_moment_ramsey(20)
    assert small["search_capped"] is False
    big = first_moment_ramsey(30)
    assert big["search_capped"] is True
    assert any("CEILING" in w for w in big.warnings)


# --------------------------------------------------------------------
# Alteration
# --------------------------------------------------------------------

def test_alteration_beats_the_union_bound_for_moderate_k():
    for k in (8, 10, 15, 20):
        a = alteration_ramsey(k)
        assert a["bound"] > a["first_moment_bound"]
        assert a["improvement"] > 0


def test_alteration_is_not_uniformly_better_and_reports_the_maximum():
    # the floor can cost one vertex at small k; both are valid bounds
    a4 = alteration_ramsey(4)
    assert a4["improvement"] == -1
    assert a4["bound"] == 5
    assert a4["first_moment_bound"] == 6
    assert a4["best_bound"] == 6
    a6 = alteration_ramsey(6)
    assert a6["improvement"] == 0


def test_the_best_bound_is_always_the_larger_of_the_two():
    for k in range(3, 22):
        a = alteration_ramsey(k)
        assert a["best_bound"] == max(a["bound"], a["first_moment_bound"])


def test_the_improvement_grows_with_k():
    imps = [alteration_ramsey(k)["improvement"] for k in (8, 10, 15, 20)]
    assert all(x < y for x, y in zip(imps, imps[1:]))


def test_alteration_bounds_lie_below_the_truth_where_known():
    for k in (3, 4):
        assert alteration_ramsey(k)["best_bound"] < \
            ramsey_number(k, k)["value"]


# --------------------------------------------------------------------
# Lovasz Local Lemma
# --------------------------------------------------------------------

def test_the_local_lemma_condition_is_e_p_d_plus_one():
    out = lovasz_local_lemma(0.01, 20)
    assert out["condition_value"] == pytest.approx(math.e * 0.01 * 21)
    assert out["applies"] is True
    assert lovasz_local_lemma(0.1, 20)["applies"] is False


def test_the_local_lemma_succeeds_where_the_union_bound_cannot():
    # the union bound scales with the NUMBER of events; the Local Lemma
    # does not care how many there are, only how entangled each is
    p, d, m = 0.01, 20, 100_000
    assert union_bound_exists(m, p)["exists"] is False
    assert lovasz_local_lemma(p, d)["applies"] is True


def test_the_reported_maximum_degree_is_the_largest_that_works():
    for p in (0.001, 0.01, 0.05):
        out = lovasz_local_lemma(p, 1)
        d_max = out["max_degree_at_p"]
        assert lovasz_local_lemma(p, d_max)["applies"] is True
        assert lovasz_local_lemma(p, d_max + 1)["applies"] is False


def test_the_reported_maximum_probability_is_the_boundary():
    for d in (5, 20, 100):
        p_max = lovasz_local_lemma(0.001, d)["max_probability_at_d"]
        assert lovasz_local_lemma(p_max, d)["applies"] is True
        assert lovasz_local_lemma(p_max * 1.001, d)["applies"] is False


def test_local_lemma_validation():
    with pytest.raises(ValueError, match="p must lie"):
        lovasz_local_lemma(1.5, 3)
    with pytest.raises(ValueError, match="non-negative"):
        lovasz_local_lemma(0.1, -1)
    with pytest.raises(ValueError, match="symmetric"):
        lovasz_local_lemma(0.1, 3, symmetric=False)


# --------------------------------------------------------------------
# Chernoff
# --------------------------------------------------------------------

def test_the_upper_chernoff_bound_is_never_violated():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(10, 200)
        p = rng.random() * 0.9 + 0.05
        t = n * p * (1 + rng.random() * 1.5)
        out = chernoff_bound(n, p, t, "upper")
        assert out["holds"] is True
        assert out["bound"] >= out["exact_tail"] - 1e-12


def test_the_lower_chernoff_bound_is_never_violated():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(10, 200)
        p = rng.random() * 0.9 + 0.05
        t = n * p * rng.random() * 0.95
        out = chernoff_bound(n, p, t, "lower")
        assert out["holds"] is True


def test_the_bound_is_not_vacuous_at_a_real_deviation():
    out = chernoff_bound(100, 0.5, 70)
    assert out["vacuous"] is False
    assert out["bound"] < 0.05
    assert out["exact_tail"] < out["bound"]


def test_a_vacuous_bound_is_flagged():
    out = chernoff_bound(100, 0.5, 50)      # the mean itself
    assert out["vacuous"] is True
    assert any("says nothing" in w for w in out.warnings)


def test_the_bound_tightens_as_the_deviation_grows():
    bounds = [chernoff_bound(200, 0.5, 100 * (1 + d))["bound"]
              for d in (0.1, 0.2, 0.3, 0.4)]
    assert all(x > y for x, y in zip(bounds, bounds[1:]))


def test_chernoff_validation():
    with pytest.raises(ValueError, match="must be positive"):
        chernoff_bound(0, 0.5, 1)
    with pytest.raises(ValueError, match="p must lie"):
        chernoff_bound(10, 2.0, 1)
    with pytest.raises(ValueError, match='tail must be'):
        chernoff_bound(10, 0.5, 6, tail="both")


# --------------------------------------------------------------------
# Azuma
# --------------------------------------------------------------------

def test_azuma_bounds_a_simulated_random_walk():
    # a +/-1 walk is a martingale with c = 1; the bound must hold
    rng = random.Random(2)
    n, reps = 100, 8000
    walks = [sum(rng.choice([-1, 1]) for _ in range(n)) for _ in range(reps)]
    for t in (20, 30, 40):
        empirical = sum(abs(x) >= t for x in walks) / reps
        assert azuma_bound(n, 1.0, t)["bound"] >= empirical


def test_azuma_is_conservative_as_it_must_be():
    # it uses only the step size, so it cannot be tight for a walk whose
    # increments are independent
    rng = random.Random(3)
    n, reps, t = 100, 8000, 30
    walks = [sum(rng.choice([-1, 1]) for _ in range(n)) for _ in range(reps)]
    empirical = sum(abs(x) >= t for x in walks) / reps
    bound = azuma_bound(n, 1.0, t)["bound"]
    assert bound > empirical
    assert bound < 1.0


def test_the_bound_falls_as_the_deviation_grows():
    bounds = [azuma_bound(100, 1.0, t)["bound"] for t in (10, 20, 30, 40)]
    assert all(x > y for x, y in zip(bounds, bounds[1:]))


def test_the_typical_deviation_is_c_root_n():
    out = azuma_bound(100, 2.0, 40)
    assert out["typical_deviation"] == pytest.approx(20.0)
    assert out["deviations_out"] == pytest.approx(2.0)


def test_azuma_validation():
    with pytest.raises(ValueError, match="n must be positive"):
        azuma_bound(0, 1.0, 1.0)
    with pytest.raises(ValueError, match="c must be positive"):
        azuma_bound(10, 0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        azuma_bound(10, 1.0, -1.0)


# --------------------------------------------------------------------
# Second moment
# --------------------------------------------------------------------

def test_the_second_moment_bound_is_variance_over_expectation_squared():
    out = second_moment_threshold(100.0, 50.0)
    assert out["p_zero_bound"] == pytest.approx(50.0 / 10000.0)
    assert out["positive_whp"] is True


def test_a_large_relative_variance_gives_nothing():
    out = second_moment_threshold(10.0, 200.0)
    assert out["p_zero_bound"] == 1.0
    assert out["vacuous"] is True
    assert out["positive_whp"] is False


def test_the_bound_never_exceeds_one():
    for e, v in [(1.0, 1e6), (10.0, 1000.0), (100.0, 1.0)]:
        assert second_moment_threshold(e, v)["p_zero_bound"] <= 1.0


def test_the_two_moments_are_complementary():
    # the first moment kills the property below threshold, the second
    # produces it above; neither alone establishes a threshold
    below = union_bound_exists(1000, 0.0005)
    assert below["exists"] is True
    above = second_moment_threshold(1000.0, 1000.0)
    assert above["positive_whp"] is True


def test_second_moment_validation():
    with pytest.raises(ValueError, match="variance must be non-negative"):
        second_moment_threshold(1.0, -1.0)
    with pytest.raises(ValueError, match="expectation must be positive"):
        second_moment_threshold(0.0, 1.0)
