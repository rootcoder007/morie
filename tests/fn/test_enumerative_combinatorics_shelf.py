"""Enumerative combinatorics, checked against the objects themselves.

Every count is verified by brute-force enumeration where that is
feasible, and against an independent recurrence or closed form where it
is not. Sources: Stanley RP (2011) *Enumerative Combinatorics* vol. 1,
2nd ed., Sec 1.9 (twelvefold way, Stirling) and 3.7 (Mobius); Andrews
GE (1976) *The Theory of Partitions* (pentagonal number theorem).
"""

import math
from itertools import permutations, product

import pytest

from morie.fn.bigint import (
    below_double_threshold,
    big_binomial,
    big_digits,
    big_factorial,
    exact_or_flag,
    fits_double,
)
from morie.fn.enumcb import (
    bell_number,
    catalan_number,
    derangements,
    mobius_inversion,
    partition_count,
    partitions_into_parts,
    stirling_first,
    stirling_second,
    twelvefold_way,
)


def set_partitions(coll):
    if not coll:
        yield []
        return
    first, rest = coll[0], coll[1:]
    for smaller in set_partitions(rest):
        for i, sub in enumerate(smaller):
            yield smaller[:i] + [[first] + sub] + smaller[i + 1:]
        yield [[first]] + smaller


def n_cycles(p):
    seen, c = set(), 0
    for i in range(len(p)):
        if i in seen:
            continue
        c += 1
        j = i
        while j not in seen:
            seen.add(j)
            j = p[j]
    return c


# --------------------------------------------------------------------
# Exact integers
# --------------------------------------------------------------------

def test_the_double_limit_is_where_it_should_be():
    assert fits_double(2 ** 53) is True
    assert fits_double(2 ** 53 + 1) is False
    assert fits_double(-(2 ** 53)) is True
    assert below_double_threshold(2 ** 53) is True
    assert below_double_threshold(2 ** 53 + 1) is False


def test_the_threshold_rule_is_sufficient_but_not_necessary():
    # 20! is about 2.4e18, far above 2^53, and yet exactly
    # representable: it is 2^18 times an odd number, so its low-order
    # bits are already zero. Only the round trip knows.
    v = big_factorial(20)
    assert v > 2 ** 53
    assert below_double_threshold(v) is False
    assert fits_double(v) is True
    assert int(float(v)) == v


def test_binomial_is_exact_where_r_is_not():
    # base R's choose(100, 50) returns 100891344545563076171808112640
    exact = big_binomial(100, 50)
    assert exact == 100891344545564193334812497256
    assert str(exact) != "100891344545563076171808112640"
    assert not fits_double(exact)


def test_the_precision_flag_reports_the_actual_error():
    out = exact_or_flag(big_binomial(100, 50), "binomial")
    assert out["exact_as_double"] is False
    assert out["absolute_error"] > 10 ** 12
    assert out["n_digits"] == 30
    assert any("2^53" in w for w in out.warnings)


def test_small_values_are_flagged_as_exact():
    out = exact_or_flag(big_factorial(15))
    assert out["exact_as_double"] is True
    assert out["absolute_error"] == 0
    assert not out.warnings


def test_factorial_digit_counts_match_known_values():
    assert big_digits(big_factorial(100)) == 158
    assert big_factorial(20) == 2432902008176640000
    assert big_factorial(0) == 1


# --------------------------------------------------------------------
# Stirling numbers, against the partitions and permutations themselves
# --------------------------------------------------------------------

def test_stirling_second_counts_set_partitions_by_block_count():
    for n in range(1, 8):
        counts = {}
        for p in set_partitions(list(range(n))):
            counts[len(p)] = counts.get(len(p), 0) + 1
        for k in range(1, n + 1):
            assert stirling_second(n, k) == counts.get(k, 0)


def test_stirling_first_counts_permutations_by_cycle_count():
    for n in range(1, 8):
        cc = {}
        for p in permutations(range(n)):
            k = n_cycles(p)
            cc[k] = cc.get(k, 0) + 1
        for k in range(1, n + 1):
            assert stirling_first(n, k) == cc.get(k, 0)


def test_stirling_first_row_sums_to_n_factorial():
    for n in range(1, 12):
        assert sum(stirling_first(n)) == math.factorial(n)


def test_signed_stirling_first_alternates():
    for n in range(1, 8):
        for k in range(n + 1):
            s = stirling_first(n, k, signed=True)
            assert abs(s) == stirling_first(n, k)
            if stirling_first(n, k) and (n - k) % 2:
                assert s < 0


def test_the_two_stirling_matrices_are_mutually_inverse():
    # sum_k s(n,k) S(k,m) = [n == m]; this is the defining relation
    for n in range(1, 8):
        for m in range(1, n + 1):
            tot = sum(stirling_first(n, k, signed=True) * stirling_second(k, m)
                      for k in range(m, n + 1))
            assert tot == (1 if n == m else 0)


def test_stirling_edge_cases():
    assert stirling_second(0, 0) == 1
    assert stirling_second(5, 0) == 0
    assert stirling_second(5, 5) == 1
    assert stirling_second(5, 1) == 1
    assert stirling_second(3, 9) == 0
    assert stirling_first(0, 0) == 1


# --------------------------------------------------------------------
# Bell numbers
# --------------------------------------------------------------------

def test_bell_counts_all_set_partitions():
    for n in range(1, 8):
        assert bell_number(n) == sum(1 for _ in set_partitions(list(range(n))))


def test_bell_is_the_row_sum_of_stirling_second():
    for n in range(30):
        assert bell_number(n) == sum(stirling_second(n, k)
                                     for k in range(n + 1))


def test_the_first_bell_numbers_are_the_known_sequence():
    assert [bell_number(i) for i in range(11)] == \
        [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]


def test_bell_stays_exact_beyond_the_double_range():
    b = bell_number(25)
    assert b == 4638590332229999353
    assert not fits_double(b)


# --------------------------------------------------------------------
# Catalan numbers
# --------------------------------------------------------------------

def balanced_bracket_count(n):
    c = 0
    for bits in product([0, 1], repeat=2 * n):
        if sum(bits) != n:
            continue
        d, ok = 0, True
        for b in bits:
            d += 1 if b else -1
            if d < 0:
                ok = False
                break
        if ok and d == 0:
            c += 1
    return c


def test_catalan_counts_balanced_brackets():
    for n in range(8):
        assert catalan_number(n) == balanced_bracket_count(n)


def test_catalan_matches_the_closed_form():
    for n in range(20):
        assert catalan_number(n) == math.comb(2 * n, n) // (n + 1)


def test_catalan_satisfies_its_own_recurrence():
    # C_{n+1} = sum_i C_i C_{n-i}
    for n in range(12):
        assert catalan_number(n + 1) == sum(
            catalan_number(i) * catalan_number(n - i) for i in range(n + 1)
        )


# --------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------

def test_the_first_partition_numbers_are_the_known_sequence():
    assert [partition_count(i) for i in range(11)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]


def test_partition_count_matches_published_landmarks():
    assert partition_count(50) == 204226
    assert partition_count(100) == 190569292
    assert partition_count(200) == 3972999029388


def test_partitions_stay_exact_far_beyond_the_double_range():
    p = partition_count(1000)
    assert p == 24061467864032622473692149727991
    assert not fits_double(p)
    assert big_digits(p) == 32


def test_eulers_theorem_distinct_equals_odd():
    # a theorem, so this must hold at every n, not on average
    for n in range(60):
        assert partition_count(n, distinct=True) == \
            partition_count(n, odd_only=True)


def test_partitions_by_part_count_sum_to_the_total():
    for n in range(1, 40):
        assert sum(partitions_into_parts(n, k) for k in range(1, n + 1)) == \
            partition_count(n)


def test_partitions_into_parts_edge_cases():
    assert partitions_into_parts(0, 0) == 1
    assert partitions_into_parts(5, 0) == 0
    assert partitions_into_parts(5, 5) == 1
    assert partitions_into_parts(5, 6) == 0
    assert partitions_into_parts(7, 3) == 4


def test_partition_validation():
    with pytest.raises(ValueError, match="non-negative"):
        partition_count(-1)
    with pytest.raises(ValueError, match="alternatives"):
        partition_count(5, distinct=True, odd_only=True)


# --------------------------------------------------------------------
# Derangements
# --------------------------------------------------------------------

def test_derangements_count_fixed_point_free_permutations():
    for n in range(8):
        brute = sum(1 for p in permutations(range(n))
                    if all(p[i] != i for i in range(n)))
        assert derangements(n) == brute


def test_derangements_match_the_inclusion_exclusion_form():
    for n in range(1, 15):
        ie = sum((-1) ** i * math.factorial(n) // math.factorial(i)
                 for i in range(n + 1))
        assert derangements(n) == ie


def test_the_derangement_ratio_approaches_one_over_e():
    r = derangements(20) / math.factorial(20)
    assert abs(r - 1 / math.e) < 1e-15


# --------------------------------------------------------------------
# Twelvefold way
# --------------------------------------------------------------------

def test_labelled_cells_against_direct_enumeration():
    for n in range(5):
        for k in range(1, 5):
            assert twelvefold_way(n, k)["count"] == k ** n
            surj = sum(1 for f in product(range(k), repeat=n)
                       if len(set(f)) == k)
            assert twelvefold_way(n, k, condition="surjective")["count"] == \
                surj
            inj = sum(1 for f in product(range(k), repeat=n)
                      if len(set(f)) == n)
            assert twelvefold_way(n, k, condition="injective")["count"] == inj


def test_unlabelled_balls_count_multisets():
    for n in range(6):
        for k in range(1, 5):
            # multisets of size n from k boxes
            got = twelvefold_way(n, k, balls="unlabelled")["count"]
            assert got == math.comb(n + k - 1, n)


def test_unlabelled_boxes_surjective_is_stirling_or_partitions():
    for n in range(1, 7):
        for k in range(1, n + 1):
            assert twelvefold_way(n, k, boxes="unlabelled",
                                  condition="surjective")["count"] == \
                stirling_second(n, k)
            assert twelvefold_way(n, k, balls="unlabelled",
                                  boxes="unlabelled",
                                  condition="surjective")["count"] == \
                partitions_into_parts(n, k)


def test_every_cell_reports_its_formula_and_is_non_negative():
    seen = set()
    for b in ("labelled", "unlabelled"):
        for x in ("labelled", "unlabelled"):
            for c in ("any", "injective", "surjective"):
                out = twelvefold_way(4, 3, balls=b, boxes=x, condition=c)
                assert out["count"] >= 0
                assert out["formula"]
                seen.add(out["cell"])
    assert len(seen) == 12


def test_twelvefold_validation():
    with pytest.raises(ValueError, match="balls must be"):
        twelvefold_way(3, 2, balls="fuzzy")
    with pytest.raises(ValueError, match="condition must be"):
        twelvefold_way(3, 2, condition="bijective")
    with pytest.raises(ValueError, match="non-negative"):
        twelvefold_way(-1, 2)


# --------------------------------------------------------------------
# Mobius inversion
# --------------------------------------------------------------------

def test_mobius_inverts_the_divisor_sum_exactly():
    g = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    f = [sum(g[d - 1] for d in range(1, m + 1) if m % d == 0)
         for m in range(1, len(g) + 1)]
    out = mobius_inversion(f)
    assert out["g"] == g
    assert out["reconstruction_residual"] == 0


def test_the_mobius_identity_holds_exactly():
    # sum_{d | n} mu(d) is 1 at n = 1 and 0 thereafter
    out = mobius_inversion([1] * 40)
    assert out["mobius_identity_residual"] == 0
    assert out["divisor_sums"][0] == 1
    assert all(s == 0 for s in out["divisor_sums"][1:])


def test_the_mobius_function_takes_its_known_values():
    out = mobius_inversion([1] * 12)
    assert out["mobius"] == [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0]


def test_inverting_the_all_ones_divisor_count_gives_the_indicator():
    # f(n) = number of divisors  =>  g(n) = 1 for all n
    f = [sum(1 for d in range(1, m + 1) if m % d == 0) for m in range(1, 21)]
    assert mobius_inversion(f)["g"] == [1] * 20


def test_mobius_validation():
    with pytest.raises(ValueError, match="must not be empty"):
        mobius_inversion([])
