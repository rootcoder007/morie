"""Extremal combinatorics and design theory, verified by construction.

Every bound is checked against an explicit construction that attains
it, and for small cases against exhaustive search over all graphs or
all families. Sources: Turan (1941); Mantel (1907); Sperner (1928);
Erdos, Ko and Rado (1961); Dilworth (1950); Fisher (1940); Kirkman
(1847); Bose (1939); Hamming (1950); Singleton (1964).
"""

import itertools
import math

import pytest

from morie.fn.dsgnth import (
    are_orthogonal,
    bibd_parameters,
    hamming_bound,
    incidence_matrix_check,
    is_latin_square,
    latin_square,
    singleton_bound,
    steiner_triple_system,
)
from morie.fn.extrgt import (
    count_edges,
    dilworth_decomposition,
    erdos_ko_rado,
    has_clique,
    mantel_number,
    sperner_width,
    turan_graph,
    turan_number,
)

FANO = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6],
        [2, 4, 5]]


def all_graphs(n):
    edges = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(edges)):
        A = [[0] * n for _ in range(n)]
        for b, (i, j) in enumerate(edges):
            if mask >> b & 1:
                A[i][j] = A[j][i] = 1
        yield A


# --------------------------------------------------------------------
# Turan and Mantel
# --------------------------------------------------------------------

def test_turan_matches_exhaustive_search_over_all_graphs():
    for n in range(2, 7):
        for r in (2, 3):
            best = max(count_edges(A) for A in all_graphs(n)
                       if has_clique(A, r + 1) is None)
            assert turan_number(n, r)["count"] == best


def test_the_turan_construction_attains_the_bound():
    for n in range(1, 30):
        for r in range(1, 6):
            g = turan_graph(n, r)
            assert g["edges"] == turan_number(n, r)["count"]
            assert turan_number(n, r)["attained"] is True


def test_the_turan_construction_has_no_forbidden_clique():
    for n in range(1, 16):
        for r in range(1, 5):
            g = turan_graph(n, r)
            assert has_clique(g["adjacency"], r + 1) is None


def test_the_rounded_formula_is_only_exact_when_r_divides_n():
    exact = turan_number(9, 3)
    assert exact["formula_is_exact"] is True
    loose = turan_number(10, 3)
    assert loose["formula_is_exact"] is False
    assert loose["count"] == 33
    assert loose["rounded_formula"] == pytest.approx(33.333, abs=0.01)


def test_mantel_is_floor_n_squared_over_four():
    for n in range(60):
        assert mantel_number(n)["count"] == n * n // 4


def test_mantel_is_turan_at_r_equals_two():
    for n in range(20):
        assert mantel_number(n)["count"] == turan_number(n, 2)["count"]


def test_turan_validation():
    with pytest.raises(ValueError, match="non-negative"):
        turan_number(-1, 2)
    with pytest.raises(ValueError, match="at least 1"):
        turan_number(5, 0)


# --------------------------------------------------------------------
# Sperner and Erdos-Ko-Rado
# --------------------------------------------------------------------

def test_sperner_width_is_the_middle_binomial():
    for n in range(12):
        assert sperner_width(n)["count"] == math.comb(n, n // 2)


def test_the_middle_layer_really_is_an_antichain():
    for n in range(2, 8):
        layer = [frozenset(c) for c in
                 itertools.combinations(range(n), n // 2)]
        assert len(layer) == sperner_width(n)["count"]
        for a, b in itertools.combinations(layer, 2):
            assert not (a < b or b < a)


def test_sperner_uniqueness_is_reported_by_parity():
    assert sperner_width(4)["unique_extremal"] is True
    assert sperner_width(5)["unique_extremal"] is False
    assert sperner_width(5)["extremal_layers"] == [2, 3]


def test_ekr_matches_exhaustive_search():
    def max_intersecting(n, k):
        sets = [frozenset(c) for c in itertools.combinations(range(n), k)]
        m = len(sets)
        best = 0
        for mask in range(1 << m):
            chosen = [sets[i] for i in range(m) if mask >> i & 1]
            if len(chosen) <= best:
                continue
            if all(a & b for a, b in itertools.combinations(chosen, 2)):
                best = len(chosen)
        return best

    for n, k in [(4, 2), (5, 2), (6, 2), (5, 3), (6, 3)]:
        assert erdos_ko_rado(n, k)["count"] == max_intersecting(n, k)


def test_ekr_is_the_star_in_its_own_regime():
    for n, k in [(6, 3), (8, 3), (10, 4), (20, 5)]:
        out = erdos_ko_rado(n, k)
        assert out["ekr_regime"] is True
        assert out["count"] == math.comb(n - 1, k - 1)


def test_below_the_regime_every_family_is_intersecting():
    # quoting C(n-1, k-1) here would be too small and look reasonable
    out = erdos_ko_rado(5, 3)
    assert out["ekr_regime"] is False
    assert out["count"] == math.comb(5, 3) == 10
    assert out["star_size"] == math.comb(4, 2) == 6
    assert out["count"] > out["star_size"]
    assert any("below 2k" in w for w in out.warnings)


def test_ekr_validation():
    with pytest.raises(ValueError, match="must not exceed n"):
        erdos_ko_rado(3, 5)


# --------------------------------------------------------------------
# Dilworth
# --------------------------------------------------------------------

def divisibility_poset(n):
    return [[(j + 1) % (i + 1) == 0 for j in range(n)] for i in range(n)]


def test_dilworth_equality_holds_on_the_divisibility_poset():
    for n in (6, 8, 10, 12):
        out = dilworth_decomposition(divisibility_poset(n))
        assert out["dilworth_holds"] is True
        assert out["antichain_size"] == out["chain_cover_size"]


def test_the_antichain_on_one_to_eight_is_the_primes_above_four():
    out = dilworth_decomposition(divisibility_poset(8))
    assert out["antichain_size"] == 4
    assert sorted(i + 1 for i in out["antichain"]) == [2, 3, 5, 7]


def test_a_chain_has_antichain_one():
    n = 6
    leq = [[j >= i for j in range(n)] for i in range(n)]
    out = dilworth_decomposition(leq)
    assert out["antichain_size"] == 1
    assert out["chain_cover_size"] == 1


def test_an_antichain_needs_one_chain_each():
    n = 5
    leq = [[i == j for j in range(n)] for i in range(n)]
    out = dilworth_decomposition(leq)
    assert out["antichain_size"] == n
    assert out["chain_cover_size"] == n


def test_dilworth_rejects_a_relation_that_is_not_a_partial_order():
    with pytest.raises(ValueError, match="reflexive"):
        dilworth_decomposition([[False, True], [False, True]])
    with pytest.raises(ValueError, match="antisymmetric"):
        dilworth_decomposition([[True, True], [True, True]])
    with pytest.raises(ValueError, match="transitive"):
        leq = [[True, True, False], [False, True, True],
               [False, False, True]]
        dilworth_decomposition(leq)


# --------------------------------------------------------------------
# BIBD
# --------------------------------------------------------------------

def test_the_fano_plane_verifies_as_a_bibd():
    c = incidence_matrix_check(FANO, 7)
    assert c["is_bibd"] is True
    assert (c["k"], c["r"], c["lambda"], c["b"]) == (3, 3, 1, 7)


def test_the_predicted_parameters_match_the_actual_fano_plane():
    p = bibd_parameters(7, 3, 1)
    c = incidence_matrix_check(FANO, 7)
    assert (p["r"], p["b"]) == (c["r"], c["b"])
    assert p["feasible"] is True


def test_the_counting_conditions_rule_out_impossible_parameters():
    assert bibd_parameters(8, 3, 1)["feasible"] is False
    assert bibd_parameters(8, 3, 1)["exists"] is False


def test_feasible_is_not_the_same_claim_as_exists():
    # (22, 7, 2) passes every counting condition and cannot exist
    out = bibd_parameters(22, 7, 2)
    assert out["divisibility_ok"] is True
    assert out["fisher_ok"] is True
    assert out["feasible"] is True
    assert out["exists"] is False
    assert any("Bruck-Ryser-Chowla" in w for w in out.warnings)


def test_an_undetermined_case_says_so_rather_than_claiming_existence():
    out = bibd_parameters(13, 4, 1)
    assert out["feasible"] is True
    assert out["exists"] is None
    assert any("NECESSARY, not sufficient" in w for w in out.warnings)


def test_fisher_inequality_is_checked():
    out = bibd_parameters(7, 3, 1)
    assert out["b"] >= out["v"]
    assert out["fisher_ok"] is True


def test_an_incomplete_design_is_rejected():
    out = incidence_matrix_check(FANO[:5], 7)
    assert out["is_bibd"] is False
    assert out["uncovered_pairs"]
    assert any("no block" in w for w in out.warnings)


def test_uneven_block_sizes_are_rejected():
    out = incidence_matrix_check([[0, 1, 2], [0, 1]], 3)
    assert out["is_bibd"] is False
    assert any("differing sizes" in w for w in out.warnings)


# --------------------------------------------------------------------
# Steiner triple systems
# --------------------------------------------------------------------

def test_existence_follows_v_mod_six():
    for v in range(3, 40):
        expected = v % 6 in (1, 3)
        assert steiner_triple_system(v, construct=False)["exists"] == expected


def test_the_triple_count_is_v_choose_two_over_three():
    for v in (7, 9, 13, 15, 19, 21):
        out = steiner_triple_system(v, construct=False)
        assert out["n_triples"] == math.comb(v, 2) // 3


def test_the_bose_construction_covers_every_pair_exactly_once():
    for v in (9, 15, 21, 27):
        out = steiner_triple_system(v)
        assert out["verified"] is True
        assert len(out["triples"]) == out["n_triples"]
        seen = {}
        for t in out["triples"]:
            for p in itertools.combinations(sorted(t), 2):
                seen[p] = seen.get(p, 0) + 1
        assert len(seen) == math.comb(v, 2)
        assert set(seen.values()) == {1}


def test_a_constructed_system_verifies_as_a_bibd():
    out = steiner_triple_system(9)
    c = incidence_matrix_check(out["triples"], 9)
    assert c["is_bibd"] is True
    assert (c["k"], c["lambda"]) == (3, 1)


def test_the_steiner_condition_is_sufficient_unlike_the_general_case():
    assert steiner_triple_system(7)["condition_is_sufficient"] is True


# --------------------------------------------------------------------
# Latin squares
# --------------------------------------------------------------------

def test_the_cyclic_construction_is_latin_at_every_order():
    for n in range(1, 12):
        assert latin_square(n)["valid"] is True
        assert is_latin_square(latin_square(n)["square"])["valid"] is True


def test_a_non_latin_grid_is_rejected():
    assert is_latin_square([[0, 0], [1, 1]])["valid"] is False
    assert is_latin_square([[0, 1], [0, 1]])["columns_ok"] is False


def test_orthogonality_requires_both_squares_to_be_latin():
    # the pair condition can hold on a grid that is not a Latin square,
    # in which case orthogonality is not a meaningful claim
    A = latin_square(4)["square"]
    B = latin_square(4, method="shifted")["square"]
    out = are_orthogonal(A, B)
    assert is_latin_square(B)["valid"] is False
    assert out["pair_condition_holds"] is True
    assert out["both_are_latin"] is False
    assert out["orthogonal"] is False


def test_a_genuine_orthogonal_pair_is_recognised():
    for n in (3, 5, 7):
        A = latin_square(n)["square"]
        B = latin_square(n, method="shifted")["square"]
        out = are_orthogonal(A, B)
        assert out["both_are_latin"] is True
        assert out["orthogonal"] is True


def test_no_orthogonal_pair_exists_at_order_two():
    # Euler was right here, by exhaustion over both Latin squares
    squares = []
    for rows in itertools.product(list(itertools.permutations(range(2))),
                                  repeat=2):
        L = [list(r) for r in rows]
        if is_latin_square(L)["valid"]:
            squares.append(L)
    assert len(squares) == 2
    assert not any(are_orthogonal(a, b)["orthogonal"]
                   for a in squares for b in squares)


def test_latin_square_validation():
    with pytest.raises(ValueError, match="at least 1"):
        latin_square(0)
    with pytest.raises(ValueError, match='method must be'):
        latin_square(4, method="magic")
    with pytest.raises(ValueError, match="same order"):
        are_orthogonal([[0, 1], [1, 0]], [[0]])


# --------------------------------------------------------------------
# Coding bounds
# --------------------------------------------------------------------

def test_the_hamming_bound_on_the_classical_perfect_codes():
    # the [7,4] Hamming code has 2^4 = 16 words and meets the bound
    assert hamming_bound(7, 3)["bound"] == 16
    assert hamming_bound(7, 3)["is_perfect_possible"] is True
    # the binary Golay code has 2^12 = 4096
    assert hamming_bound(23, 7)["bound"] == 4096
    # the ternary Golay code has 3^6 = 729
    assert hamming_bound(11, 5, q=3)["bound"] == 729


def test_the_ball_volume_is_the_sum_of_binomials():
    for n, d in [(7, 3), (15, 3), (23, 7)]:
        t = (d - 1) // 2
        vol = sum(math.comb(n, i) for i in range(t + 1))
        assert hamming_bound(n, d)["ball_volume"] == vol


def test_a_non_perfect_case_is_flagged_as_such():
    out = hamming_bound(5, 3)
    assert out["is_perfect_possible"] is False
    assert out["bound"] == 5


def test_the_singleton_bound_and_which_is_tighter():
    assert singleton_bound(7, 3)["bound"] == 2 ** 5
    out = singleton_bound(23, 7)
    assert out["hamming_is_tighter"] is True
    assert out["tighter"] == out["hamming_bound"]


def test_repetition_code_saturates_singleton():
    # the length-n binary repetition code has d = n and exactly 2 words
    for n in range(2, 8):
        assert singleton_bound(n, n)["bound"] == 2


def test_coding_bound_validation():
    with pytest.raises(ValueError, match="must not exceed n"):
        hamming_bound(3, 5)
    with pytest.raises(ValueError, match="at least 2"):
        hamming_bound(7, 3, q=1)
    with pytest.raises(ValueError, match="must not exceed n"):
        singleton_bound(3, 5)
