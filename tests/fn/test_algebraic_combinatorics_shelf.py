"""Algebraic combinatorics, checked against enumeration and bijectivity.

Two of these are bijections rather than formulas, which admits a
stronger test than a count: RSK is round-tripped over every permutation
of n <= 6, and its corollary sum (f^lambda)^2 = n! is checked as an
identity. The hook length formula is checked against brute-force
enumeration of standard tableaux, and Burnside against orbits counted
directly.

Sources: Sagan (2001) *The Symmetric Group* 2nd ed.; Frame, Robinson
and Thrall (1954); Schensted (1961); Knuth (1970); Burnside (1897);
Polya (1937).
"""

import itertools
import math

import pytest

from morie.fn.algcmb import (
    burnside_orbit_count,
    cycle_index_necklaces,
    hook_lengths,
    partitions_of,
    rsk_correspondence,
    rsk_insert,
    rsk_inverse,
    standard_tableaux_count,
)


def brute_force_syt(shape):
    """Count standard Young tableaux by filling every way."""
    n = sum(shape)
    cells = [(i, j) for i, r in enumerate(shape) for j in range(r)]
    count = 0
    for perm in itertools.permutations(range(1, n + 1)):
        T = dict(zip(cells, perm))
        ok = True
        for (i, j), v in T.items():
            if (i, j + 1) in T and T[(i, j + 1)] < v:
                ok = False
                break
            if (i + 1, j) in T and T[(i + 1, j)] < v:
                ok = False
                break
        if ok:
            count += 1
    return count


def longest_monotone(w, increasing=True):
    best = 0
    for r in range(1, len(w) + 1):
        for c in itertools.combinations(w, r):
            pairs = zip(c, c[1:])
            if all(a < b for a, b in pairs) if increasing else \
                    all(a > b for a, b in zip(c, c[1:])):
                best = max(best, r)
    return best


def direct_orbits(G, k, n):
    seen, orbits = set(), 0
    for col in itertools.product(range(k), repeat=n):
        if col in seen:
            continue
        orbits += 1
        for g in G:
            seen.add(tuple(col[g[i]] for i in range(n)))
    return orbits


def rotations(n):
    return [[(i + s) % n for i in range(n)] for s in range(n)]


# --------------------------------------------------------------------
# Partitions and hooks
# --------------------------------------------------------------------

def test_partitions_of_matches_the_known_counts():
    assert [len(partitions_of(n)) for n in range(9)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22]


def test_every_partition_is_weakly_decreasing_and_sums_right():
    for n in range(1, 9):
        for p in partitions_of(n):
            assert sum(p) == n
            assert all(a >= b for a, b in zip(p, p[1:]))


def test_hook_lengths_on_a_worked_shape():
    h = hook_lengths((3, 2))
    assert h["hooks"] == [[4, 3, 1], [2, 1]]
    assert h["product"] == 24
    assert h["conjugate"] == (2, 2, 1)


def test_the_hook_of_a_single_row_counts_down():
    assert hook_lengths((5,))["hooks"] == [[5, 4, 3, 2, 1]]


def test_the_hook_of_a_single_column_counts_down():
    assert hook_lengths((1, 1, 1))["hooks"] == [[3], [2], [1]]


def test_hook_lengths_reject_a_non_partition():
    with pytest.raises(ValueError, match="weakly decreasing"):
        hook_lengths((2, 3))
    with pytest.raises(ValueError, match="must be positive"):
        hook_lengths((3, 0))


# --------------------------------------------------------------------
# Hook length formula against brute force
# --------------------------------------------------------------------

def test_the_hook_formula_matches_brute_force_enumeration():
    for n in range(1, 8):
        for shape in partitions_of(n):
            assert standard_tableaux_count(shape)["count"] == \
                brute_force_syt(shape)


def test_the_hook_product_always_divides_n_factorial():
    for n in range(1, 11):
        for shape in partitions_of(n):
            out = standard_tableaux_count(shape)
            assert out["remainder"] == 0
            assert out["divides_exactly"] is True
            assert not out.warnings


def test_a_single_row_or_column_has_exactly_one_tableau():
    for n in range(1, 8):
        assert standard_tableaux_count((n,))["count"] == 1
        assert standard_tableaux_count(tuple([1] * n))["count"] == 1


def test_the_hook_formula_on_known_values():
    assert standard_tableaux_count((3, 2))["count"] == 5
    assert standard_tableaux_count((2, 2))["count"] == 2
    assert standard_tableaux_count((2, 1))["count"] == 2
    assert standard_tableaux_count((3, 2, 1))["count"] == 16


# --------------------------------------------------------------------
# RSK: a bijection, so round-trip it
# --------------------------------------------------------------------

def test_rsk_round_trips_every_permutation():
    for n in range(1, 7):
        for w in itertools.permutations(range(1, n + 1)):
            out = rsk_correspondence(list(w))
            assert rsk_inverse(out["p_tableau"], out["q_tableau"]) == list(w)


def test_p_and_q_always_share_a_shape():
    for n in range(1, 7):
        for w in itertools.permutations(range(1, n + 1)):
            assert rsk_correspondence(list(w))["same_shape"] is True


def test_the_rsk_corollary_sum_of_squares_is_n_factorial():
    # every permutation maps to a distinct (P, Q) pair of the same
    # shape, so the squares of the tableau counts must total n!
    for n in range(1, 9):
        total = sum(standard_tableaux_count(s)["count"] ** 2
                    for s in partitions_of(n))
        assert total == math.factorial(n)


def test_the_shape_gives_the_longest_monotone_subsequences():
    for n in (5, 6):
        for w in itertools.permutations(range(1, n + 1)):
            out = rsk_correspondence(list(w))
            assert out["longest_increasing"] == longest_monotone(w, True)
            assert out["longest_decreasing"] == longest_monotone(w, False)


def test_the_identity_permutation_gives_a_single_row():
    out = rsk_correspondence([1, 2, 3, 4, 5])
    assert out["shape"] == (5,)
    assert out["longest_increasing"] == 5
    assert out["longest_decreasing"] == 1


def test_the_reversed_permutation_gives_a_single_column():
    out = rsk_correspondence([5, 4, 3, 2, 1])
    assert out["shape"] == (1, 1, 1, 1, 1)
    assert out["longest_increasing"] == 1
    assert out["longest_decreasing"] == 5


def test_shapes_produced_are_partitions_of_n():
    for w in itertools.permutations(range(1, 6)):
        shape = rsk_correspondence(list(w))["shape"]
        assert sum(shape) == 5
        assert all(a >= b for a, b in zip(shape, shape[1:]))


def test_row_insertion_bumps_correctly():
    T, row = rsk_insert([[1, 3, 5]], 4)
    assert T == [[1, 3, 4], [5]]
    assert row == 1
    T2, row2 = rsk_insert([[1, 3, 5]], 6)
    assert T2 == [[1, 3, 5, 6]]
    assert row2 == 0


def test_rsk_validation():
    with pytest.raises(ValueError, match="expected a permutation"):
        rsk_correspondence([1, 1, 2])
    with pytest.raises(ValueError, match="same shape"):
        rsk_inverse([[1, 2]], [[1]])


# --------------------------------------------------------------------
# Burnside
# --------------------------------------------------------------------

def test_burnside_matches_direct_orbit_enumeration():
    for n in range(1, 7):
        G = rotations(n)
        for k in (2, 3):
            assert burnside_orbit_count(G, k)["orbits"] == \
                direct_orbits(G, k, n)


def test_dividing_by_the_group_order_is_wrong():
    # the naive "divide by the symmetry" fails whenever some
    # arrangements have symmetry of their own
    out = burnside_orbit_count(rotations(4), 2)
    assert out["orbits"] == 6
    assert out["naive_division"] == 4.0
    assert out["naive_is_wrong"] is True


def test_the_trivial_group_leaves_every_colouring_distinct():
    G = [[0, 1, 2]]
    out = burnside_orbit_count(G, 3)
    assert out["orbits"] == 27
    assert out["naive_is_wrong"] is False


def test_the_fixed_point_total_is_divisible_by_the_group_order():
    for n in range(1, 7):
        for k in (2, 3, 4):
            out = burnside_orbit_count(rotations(n), k)
            assert out["divides_exactly"] is True
            assert not out.warnings


def test_burnside_rejects_a_non_permutation():
    with pytest.raises(ValueError, match="not a permutation"):
        burnside_orbit_count([[0, 0]], 2)
    with pytest.raises(ValueError, match="same set"):
        burnside_orbit_count([[0, 1], [0, 1, 2]], 2)
    with pytest.raises(ValueError, match="at least the identity"):
        burnside_orbit_count([], 2)


# --------------------------------------------------------------------
# Necklaces
# --------------------------------------------------------------------

def test_the_closed_form_agrees_with_direct_burnside():
    for n in range(1, 13):
        for k in (2, 3, 4):
            out = cycle_index_necklaces(n, k)
            assert out["agrees"] is True
            assert out["count"] == out["direct_burnside"]


def test_two_colour_necklaces_are_the_known_sequence():
    assert [cycle_index_necklaces(n, 2)["count"] for n in range(1, 9)] == \
        [2, 3, 4, 6, 8, 14, 20, 36]


def test_three_colour_necklaces_are_the_known_sequence():
    assert [cycle_index_necklaces(n, 3)["count"] for n in range(1, 7)] == \
        [3, 6, 11, 24, 51, 130]


def test_the_cyclic_sum_always_divides_by_n():
    for n in range(1, 15):
        for k in (2, 3, 5):
            assert cycle_index_necklaces(n, k)["divides_exactly"] is True


def test_one_bead_gives_one_orbit_per_colour():
    for k in (1, 2, 5):
        assert cycle_index_necklaces(1, k)["count"] == k


def test_necklace_validation():
    with pytest.raises(ValueError, match="n must be positive"):
        cycle_index_necklaces(0, 2)
    with pytest.raises(ValueError, match="k must be positive"):
        cycle_index_necklaces(3, 0)
