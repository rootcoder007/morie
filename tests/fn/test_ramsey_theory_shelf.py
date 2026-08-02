"""Ramsey theory: exact values, certified bounds, exhaustive proofs.

F. P. Ramsey's combinatorics, not J. B. Ramsey's RESET test -- see
test_reset_consolidation below for the deliberate separation.

Sources: Radziszowski, *Small Ramsey Numbers*, EJC Dynamic Survey DS1
rev. 17 (2024), doi:10.37236/21, Tables Ia and Ib; Goodman (1959)
*Amer Math Monthly* 66:778-783; Erdos (1947) *Bull AMS* 53:292-294;
Greenwood and Gleason (1955).
"""

import itertools
import math

from morie.fn import _array_core as np
import pytest

from morie.fn.ramthy import (
    goodman_minimum,
    monochromatic_triangles,
    party_problem,
    ramsey_lower_bound_probabilistic,
    ramsey_number,
    ramsey_upper_bound,
    verify_ramsey_witness,
)


def all_colourings(n):
    """Every 2-colouring of K_n, as symmetric 0/1 matrices."""
    edges = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(edges)):
        C = np.zeros((n, n), dtype=int)
        for b, (i, j) in enumerate(edges):
            if mask >> b & 1:
                C[i, j] = C[j, i] = 1
        yield C


def cycle_colouring(n):
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        C[i, (i + 1) % n] = C[(i + 1) % n, i] = 1
    return C


# --------------------------------------------------------------------
# Known values, from the primary source
# --------------------------------------------------------------------

def test_the_nine_known_values_match_ds1_table_ia():
    known = {(3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23,
             (3, 8): 28, (3, 9): 36, (4, 4): 18, (4, 5): 25}
    for (k, l), v in known.items():
        assert ramsey_number(k, l)["value"] == v
        assert ramsey_number(k, l)["exact"] is True


def test_ramsey_numbers_are_symmetric():
    for k, l in [(3, 4), (3, 7), (4, 5), (5, 5)]:
        a, b = ramsey_number(k, l), ramsey_number(l, k)
        assert a["value"] == b["value"]
        assert (a["lower"], a["upper"]) == (b["lower"], b["upper"])


def test_the_trivial_cases_are_exact():
    assert ramsey_number(1, 9)["value"] == 1
    for l in range(2, 8):
        assert ramsey_number(2, l)["value"] == l


def test_unknown_values_return_an_interval_and_never_a_number():
    for k, l in [(5, 5), (6, 6), (4, 6), (3, 10)]:
        out = ramsey_number(k, l)
        assert out["value"] is None
        assert out["exact"] is False
        assert out["lower"] < out["upper"]
        assert any("never been determined" in w for w in out.warnings)


def test_the_circulating_wrong_value_for_r55_is_flagged():
    # DS1: "The claim that R(5,5) = 50 ... is in error, and despite
    # being shown to be incorrect more than once, this value is still
    # being cited by some authors."
    out = ramsey_number(5, 5)
    assert (out["lower"], out["upper"]) == (43, 46)
    assert out["value"] != 50
    assert any("50 is incorrect" in w for w in out.warnings)


def test_known_values_lie_inside_their_own_bounds():
    for k, l in [(3, 3), (3, 5), (4, 4), (4, 5)]:
        out = ramsey_number(k, l)
        assert out["lower"] == out["value"] == out["upper"]
        assert out["value"] <= out["erdos_szekeres_bound"]


# --------------------------------------------------------------------
# Goodman's identity -- exact, not approximate
# --------------------------------------------------------------------

def test_goodman_identity_matches_brute_force_on_random_colourings():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 11))
        C = np.triu((rng.random((n, n)) < 0.5).astype(int), 1)
        C = C + C.T
        out = monochromatic_triangles(C, brute_force=True)
        assert out["identity_residual"] == 0
        assert out["red_triangles"] + out["blue_triangles"] == out["estimate"]


def test_monochromatic_plus_bichromatic_is_every_triangle():
    rng = np.random.default_rng(1)
    for n in (4, 6, 9):
        C = np.triu((rng.random((n, n)) < 0.5).astype(int), 1)
        C = C + C.T
        out = monochromatic_triangles(C)
        assert out["monochromatic"] + out["bichromatic"] == math.comb(n, 3)


def test_an_all_red_graph_is_entirely_monochromatic():
    n = 7
    C = np.ones((n, n), dtype=int)
    np.fill_diagonal(C, 0)
    assert monochromatic_triangles(C)["estimate"] == math.comb(n, 3)


def test_the_five_cycle_has_no_monochromatic_triangle():
    assert monochromatic_triangles(cycle_colouring(5))["estimate"] == 0


def test_goodman_minimum_is_attained_exhaustively_for_small_n():
    for n in (3, 4, 5, 6):
        observed = min(monochromatic_triangles(C)["estimate"]
                       for C in all_colourings(n))
        assert observed == goodman_minimum(n)["minimum"]


def test_every_colouring_of_k6_has_at_least_two_mono_triangles():
    # the exhaustive half of R(3,3) <= 6: all 2^15 colourings
    worst = min(monochromatic_triangles(C)["estimate"]
                for C in all_colourings(6))
    assert worst == 2
    assert goodman_minimum(6)["minimum"] == 2


def test_no_colouring_ever_falls_below_the_goodman_bound():
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(3, 13))
        C = np.triu((rng.random((n, n)) < rng.random()).astype(int), 1)
        C = C + C.T
        out = monochromatic_triangles(C)
        assert out["estimate"] >= out["goodman_minimum"]
        assert not any("impossible" in w for w in out.warnings)


def test_goodman_minimum_is_zero_below_six_and_positive_from_six():
    for n in (3, 4, 5):
        assert goodman_minimum(n)["minimum"] == 0
    for n in (6, 7, 8, 9, 10):
        assert goodman_minimum(n)["minimum"] >= 1


# --------------------------------------------------------------------
# The party problem, proved rather than quoted
# --------------------------------------------------------------------

def test_six_people_force_a_monochromatic_triple():
    out = party_problem(6)
    assert out["guaranteed"] is True
    assert out["minimum_monochromatic"] == 2
    assert out["ramsey_number"] == 6


def test_five_people_do_not_and_the_witness_is_verified():
    out = party_problem(5)
    assert out["guaranteed"] is False
    assert out["witness_valid"] is True
    assert monochromatic_triangles(out["witness"])["estimate"] == 0


def test_the_five_cycle_witness_certifies_the_lower_bound():
    w = verify_ramsey_witness(cycle_colouring(5), 3, 3)
    assert w["valid"] is True
    assert w["red_clique"] is None and w["blue_clique"] is None
    assert w["certifies"] == "R(3,3) > 5"


def test_a_bad_witness_is_rejected_with_the_offending_clique():
    C = np.ones((6, 6), dtype=int)
    np.fill_diagonal(C, 0)
    w = verify_ramsey_witness(C, 3, 3)
    assert w["valid"] is False
    assert w["red_clique"] is not None
    assert len(w["red_clique"]) == 3


def test_no_colouring_of_k6_can_witness_r33():
    # the other half of R(3,3) = 6, by exhaustion over every colouring
    assert not any(verify_ramsey_witness(C, 3, 3)["valid"]
                   for C in all_colourings(6))


# --------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------

def test_the_pure_recursion_derives_the_classical_values_tightly():
    # Greenwood and Gleason: R(k-1,l) + R(k,l-1), strict when both even.
    # With use_known=False nothing is looked up, so these are derived.
    for (k, l), v in [((3, 3), 6), ((3, 4), 9), ((3, 5), 14),
                      ((4, 4), 18)]:
        b = ramsey_upper_bound(k, l, use_known=False)
        assert b["used_known_values"] is False
        assert b["recursive"] == v


def test_the_recursion_is_not_tight_everywhere():
    # R(4,5) = 25 but the recursion alone only gives 31; claiming the
    # argument reproduces every known value would be false
    b = ramsey_upper_bound(4, 5, use_known=False)
    assert b["recursive"] == 31
    assert ramsey_number(4, 5)["value"] == 25


def test_looking_up_known_values_tightens_the_recursion():
    pure = ramsey_upper_bound(5, 5, use_known=False)["recursive"]
    look = ramsey_upper_bound(5, 5, use_known=True)["recursive"]
    assert look < pure


def test_the_binomial_bound_is_weaker_than_the_recursion():
    for k, l in [(3, 4), (4, 4), (4, 5), (5, 5), (6, 6)]:
        b = ramsey_upper_bound(k, l, use_known=False)
        assert b["binomial"] >= b["recursive"]
        assert b["best"] == min(b["binomial"], b["recursive"])


def test_every_upper_bound_actually_bounds_the_known_value():
    for k, l in [(3, 3), (3, 4), (3, 5), (3, 6), (4, 4), (4, 5)]:
        v = ramsey_number(k, l)["value"]
        b = ramsey_upper_bound(k, l, use_known=False)
        assert v <= b["recursive"]
        assert v <= b["binomial"]


def test_the_probabilistic_lower_bound_is_below_every_known_value():
    for k in (3, 4):
        lb = ramsey_lower_bound_probabilistic(k)["bound"]
        assert lb < ramsey_number(k, k)["value"]


def test_the_probabilistic_bound_beats_two_to_the_k_over_two():
    # the union-bound calculation is stronger than the clean asymptotic
    # form usually quoted from it
    for k in (10, 15, 20):
        b = ramsey_lower_bound_probabilistic(k)
        assert b["bound"] > b["asymptotic_2_to_k_over_2"]


def test_the_expected_count_really_is_below_one_at_the_bound():
    for k in (4, 6, 10):
        b = ramsey_lower_bound_probabilistic(k)
        assert b["expected_at_bound"] < 1.0


def test_the_probabilistic_bound_grows_with_k():
    bounds = [ramsey_lower_bound_probabilistic(k)["bound"]
              for k in range(3, 15)]
    assert all(b <= c for b, c in zip(bounds, bounds[1:]))


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------

def test_ramsey_input_validation():
    with pytest.raises(ValueError, match="at least 1"):
        ramsey_number(0, 3)
    with pytest.raises(ValueError, match="at least three vertices"):
        monochromatic_triangles(np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="must be square"):
        monochromatic_triangles(np.zeros((3, 4), dtype=int))
    with pytest.raises(ValueError, match="symmetric"):
        C = np.zeros((4, 4), dtype=int)
        C[0, 1] = 1
        monochromatic_triangles(C)
    with pytest.raises(ValueError, match="n must be at least 3"):
        goodman_minimum(2)
    with pytest.raises(ValueError, match="k must be at least 2"):
        ramsey_lower_bound_probabilistic(1)


# --------------------------------------------------------------------
# The two Ramseys are different people
# --------------------------------------------------------------------

def test_reset_consolidation_all_four_implementations_agree():
    # J. B. Ramsey's RESET, unrelated to F. P. Ramsey above. Four
    # modules carried four copies of this arithmetic; they now share
    # one native core and must agree exactly.
    from morie.fn.ramsy import ramsey_reset_test as ramsy_f
    from morie.fn.reset import ramsey_reset_test as reset_f
    from morie.fn.rmsyt import ramsey_reset as rmsyt_f
    from morie.fn.rsetf import ramsey_reset as rsetf_f

    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([1.0, 2.0]) + 1.5 * x ** 2 + rng.normal(size=n)
    fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]

    a = reset_f(y, X, fitted).statistic
    b = ramsy_f(X[:, 1:], y).statistic
    c = rmsyt_f(y, X[:, 1:]).statistic
    d = rsetf_f(y, X)["statistic"]
    assert a == pytest.approx(b, rel=1e-9)
    assert a == pytest.approx(c, rel=1e-9)
    assert a == pytest.approx(d, rel=1e-9)


def test_reset_is_invariant_to_the_scale_of_the_response():
    # the defect that motivated the consolidation: cubing unscaled
    # fitted values made F depend on the units of y
    from morie.fn.rsetf import ramsey_reset

    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([1.0, 2.0]) + 1.5 * x ** 2 + rng.normal(size=n)
    base = ramsey_reset(y, X)["statistic"]
    for s in (1e2, 1e4, 1e6):
        assert ramsey_reset(y * s, X)["statistic"] == pytest.approx(
            base, rel=1e-6
        )
