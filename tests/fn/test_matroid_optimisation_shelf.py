"""Matroids and combinatorial optimisation, checked against optima.

The organising results are equivalences and dualities, so both sides
are computed and compared rather than one side being asserted. Rado
and Edmonds' greedy theorem is tested in BOTH directions: greedy is
optimal on matroids, and provably fails on an independence system that
is hereditary but violates exchange.

Sources: Oxley (2011) *Matroid Theory* 2nd ed.; Whitney (1935);
Edmonds (1971) *Math Prog* 1:127-136; Kruskal (1956) *Proc AMS*
7(1):48-50; Hall (1935); Konig (1931); Ford and Fulkerson (1956).
"""

import itertools
import math
import random

import pytest

from morie.fn.cmbopt import (
    bipartite_matching,
    hall_condition,
    konig_theorem,
    max_flow_min_cut,
    minimum_spanning_tree,
)
from morie.fn.matrdt import (
    brute_force_max_weight,
    graphic_matroid,
    greedy_independent_set,
    is_matroid,
    matroid_bases,
    matroid_circuits,
    matroid_dual,
    matroid_rank,
    uniform_matroid,
)

# hereditary, but {0} and {1,2} violate exchange
NON_MATROID_GROUND = [0, 1, 2]
NON_MATROID_IND = [(), (0,), (1,), (2,), (1, 2)]


# --------------------------------------------------------------------
# Matroid axioms
# --------------------------------------------------------------------

def test_uniform_matroids_satisfy_both_axioms():
    for n in range(1, 6):
        for k in range(n + 1):
            u = uniform_matroid(n, k)
            assert is_matroid(u["ground"], u["independent"])["is_matroid"]


def test_graphic_matroids_satisfy_both_axioms():
    for edges, n in [([(0, 1), (1, 2), (0, 2)], 3),
                     ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4),
                     ([(0, 1), (2, 3)], 4)]:
        g = graphic_matroid(edges, n)
        assert is_matroid(g["ground"], g["independent"])["is_matroid"]


def test_a_non_matroid_is_identified_with_its_witness():
    out = is_matroid(NON_MATROID_GROUND, NON_MATROID_IND)
    assert out["is_matroid"] is False
    assert out["hereditary"] is True
    assert out["exchange"] is False
    assert out["exchange_violation"] == ([0], [1, 2])
    assert any("Exchange fails" in w for w in out.warnings)


def test_a_non_hereditary_system_is_rejected_before_exchange():
    out = is_matroid([0, 1], [(), (0, 1)])
    assert out["hereditary"] is False
    assert out["is_matroid"] is False
    assert any("Heredity fails" in w for w in out.warnings)


def test_an_empty_family_is_not_a_matroid():
    assert is_matroid([0, 1], [])["is_matroid"] is False


# --------------------------------------------------------------------
# Rank, bases, circuits, duality
# --------------------------------------------------------------------

def test_uniform_matroid_rank_and_basis_count():
    for n in range(1, 6):
        for k in range(n + 1):
            u = uniform_matroid(n, k)
            assert matroid_rank(u["ground"], u["independent"]) == min(k, n)
            assert len(matroid_bases(u["ground"], u["independent"])) == \
                math.comb(n, min(k, n))


def test_all_bases_have_the_same_size():
    # a consequence of exchange, not an assumption
    for edges, n in [([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4),
                     ([(0, 1), (1, 2), (0, 2), (2, 3)], 4)]:
        g = graphic_matroid(edges, n)
        sizes = {len(b) for b in matroid_bases(g["ground"], g["independent"])}
        assert len(sizes) == 1


def test_uniform_matroid_circuits_are_the_k_plus_one_subsets():
    u = uniform_matroid(4, 2)
    circ = matroid_circuits(u["ground"], u["independent"])
    assert all(len(c) == 3 for c in circ)
    assert len(circ) == math.comb(4, 3)


def test_a_graphic_matroid_circuit_is_a_graph_cycle():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    g = graphic_matroid(edges, 4)
    circ = matroid_circuits(g["ground"], g["independent"])
    assert [0, 1, 2] in circ          # the triangle
    assert all(len(c) >= 3 for c in circ)


def test_duality_is_an_involution():
    for n, k in [(3, 1), (4, 2), (4, 3), (5, 2)]:
        u = uniform_matroid(n, k)
        d = matroid_dual(u["ground"], u["independent"])
        dd = matroid_dual(u["ground"], d)
        assert sorted(map(sorted, dd)) == sorted(map(sorted, u["independent"]))


def test_the_dual_of_a_uniform_matroid_has_complementary_rank():
    for n, k in [(4, 1), (4, 2), (5, 3), (6, 2)]:
        u = uniform_matroid(n, k)
        d = matroid_dual(u["ground"], u["independent"])
        assert matroid_rank(u["ground"], d) == n - k


def test_the_dual_of_a_matroid_is_a_matroid():
    for n, k in [(3, 1), (4, 2), (5, 2)]:
        u = uniform_matroid(n, k)
        d = matroid_dual(u["ground"], u["independent"])
        assert is_matroid(u["ground"], d)["is_matroid"]


# --------------------------------------------------------------------
# Rado-Edmonds, both directions
# --------------------------------------------------------------------

def test_greedy_is_optimal_on_uniform_matroids():
    rng = random.Random(0)
    for n, k in [(4, 2), (5, 2), (5, 3), (6, 3)]:
        u = uniform_matroid(n, k)
        for _ in range(40):
            w = [rng.randint(-5, 10) for _ in range(n)]
            g = greedy_independent_set(u["ground"], u["independent"], w)
            b = brute_force_max_weight(u["ground"], u["independent"], w)
            assert g["weight"] == b["weight"]


def test_greedy_is_optimal_on_graphic_matroids():
    rng = random.Random(1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    g = graphic_matroid(edges, 4)
    for _ in range(150):
        w = [rng.randint(1, 20) for _ in range(len(edges))]
        got = greedy_independent_set(g["ground"], g["independent"], w)
        best = brute_force_max_weight(g["ground"], g["independent"], w)
        assert got["weight"] == best["weight"]


def test_greedy_fails_on_a_system_that_is_not_a_matroid():
    # the converse half of Rado-Edmonds. A test that only checked the
    # forward direction would pass on code with no notion of a matroid.
    rng = random.Random(2)
    failures = 0
    for _ in range(300):
        w = [rng.randint(1, 10) for _ in range(3)]
        g = greedy_independent_set(NON_MATROID_GROUND, NON_MATROID_IND, w)
        b = brute_force_max_weight(NON_MATROID_GROUND, NON_MATROID_IND, w)
        assert g["weight"] <= b["weight"]
        if g["weight"] < b["weight"]:
            failures += 1
    assert failures > 0


def test_a_concrete_weighting_where_greedy_loses():
    w = [5, 3, 3]
    g = greedy_independent_set(NON_MATROID_GROUND, NON_MATROID_IND, w)
    b = brute_force_max_weight(NON_MATROID_GROUND, NON_MATROID_IND, w)
    assert g["set"] == [0] and g["weight"] == 5
    assert b["set"] == [1, 2] and b["weight"] == 6


def test_greedy_never_takes_a_negative_weight_element():
    u = uniform_matroid(4, 3)
    g = greedy_independent_set(u["ground"], u["independent"],
                               [-1, -2, 5, -3])
    assert g["set"] == [2]


# --------------------------------------------------------------------
# Minimum spanning tree
# --------------------------------------------------------------------

def brute_mst(edges, n, w):
    best = None
    for c in itertools.combinations(range(len(edges)), n - 1):
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        ok = True
        for i in c:
            a, b = edges[i]
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            par[ra] = rb
        if ok:
            t = sum(w[i] for i in c)
            if best is None or t < best:
                best = t
    return best


def test_mst_matches_exhaustive_search_over_spanning_trees():
    rng = random.Random(3)
    checked = 0
    for _ in range(120):
        n = rng.randint(3, 6)
        edges = [e for e in itertools.combinations(range(n), 2)
                 if rng.random() < 0.75]
        if len(edges) < n - 1:
            continue
        w = [rng.randint(1, 20) for _ in edges]
        got = minimum_spanning_tree(edges, n, w)
        if not got["connected"]:
            continue
        assert got["weight"] == brute_mst(edges, n, w)
        checked += 1
    assert checked > 30


def test_mst_is_exactly_greedy_on_the_cycle_matroid():
    # the reason Kruskal is correct, made testable
    rng = random.Random(4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    g = graphic_matroid(edges, 4)
    for _ in range(150):
        w = [rng.randint(1, 20) for _ in edges]
        mst = minimum_spanning_tree(edges, 4, w)
        M = max(w) + 1
        got = greedy_independent_set(g["ground"], g["independent"],
                                     [M - x for x in w])
        assert sum(w[i] for i in got["set"]) == mst["weight"]


def test_a_tree_has_n_minus_one_edges():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    out = minimum_spanning_tree(edges, 4, [1, 2, 3, 4, 5])
    assert out["connected"] is True
    assert out["n_edges_chosen"] == 3
    assert out["weight"] == 6


def test_a_disconnected_graph_gives_a_forest_and_says_so():
    out = minimum_spanning_tree([(0, 1), (2, 3)], 4, [1, 1])
    assert out["connected"] is False
    assert out["n_components"] == 2
    assert any("spanning forest" in w for w in out.warnings)


def test_mst_validation():
    with pytest.raises(ValueError, match="weights has length"):
        minimum_spanning_tree([(0, 1)], 2, [1, 2])
    with pytest.raises(ValueError, match="leaves 0"):
        minimum_spanning_tree([(0, 5)], 2, [1])


# --------------------------------------------------------------------
# Konig and Hall
# --------------------------------------------------------------------

def test_konig_holds_on_random_bipartite_graphs():
    rng = random.Random(5)
    for _ in range(150):
        ln, rn = rng.randint(1, 5), rng.randint(1, 5)
        E = [(a, b) for a in range(ln) for b in range(rn)
             if rng.random() < 0.5]
        out = konig_theorem(ln, rn, E)
        assert out["konig_holds"] is True
        assert out["matching_size"] == out["cover_size"]
        assert out["cover_is_valid"] is True


def test_the_konig_cover_really_covers_every_edge():
    # a cover of the right size that misses an edge would otherwise pass
    E = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)]
    out = konig_theorem(3, 3, E)
    cl, cr = set(out["cover_left"]), set(out["cover_right"])
    assert all(a in cl or b in cr for a, b in E)
    assert out["uncovered_edges"] == []


def test_matching_size_matches_brute_force():
    rng = random.Random(6)
    for _ in range(60):
        ln, rn = rng.randint(1, 4), rng.randint(1, 4)
        E = [(a, b) for a in range(ln) for b in range(rn)
             if rng.random() < 0.6]
        got = bipartite_matching(ln, rn, E)["size"]
        best = 0
        for r in range(min(ln, rn), 0, -1):
            found = False
            for sub in itertools.combinations(E, r):
                if len({a for a, _ in sub}) == r and \
                        len({b for _, b in sub}) == r:
                    found = True
                    break
            if found:
                best = r
                break
        assert got == best


def test_hall_agrees_with_the_matching_everywhere():
    rng = random.Random(7)
    for _ in range(200):
        ln, rn = rng.randint(1, 5), rng.randint(1, 5)
        E = [(a, b) for a in range(ln) for b in range(rn)
             if rng.random() < 0.5]
        assert hall_condition(ln, rn, E)["agrees_with_matching"] is True


def test_hall_returns_the_violating_set_as_a_certificate():
    # three left vertices, two of which share a single neighbour
    out = hall_condition(3, 2, [(0, 0), (1, 0), (2, 1)])
    assert out["holds"] is False
    assert out["violating_set"] == [0, 1]
    assert out["deficiency"] == 1
    assert out["perfect_on_left"] is False


def test_hall_holds_on_a_complete_bipartite_graph():
    E = [(a, b) for a in range(3) for b in range(3)]
    out = hall_condition(3, 3, E)
    assert out["holds"] is True
    assert out["matching_size"] == 3
    assert out["violating_set"] is None


# --------------------------------------------------------------------
# Max-flow min-cut
# --------------------------------------------------------------------

def test_max_flow_equals_min_cut_on_random_networks():
    rng = random.Random(8)
    for _ in range(150):
        n = rng.randint(3, 6)
        C = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.5:
                    C[i][j] = rng.randint(0, 15)
        out = max_flow_min_cut(C, 0, n - 1)
        assert out["theorem_holds"] is True
        assert out["residual_gap"] < 1e-9


def test_the_classic_network():
    C = [[0, 3, 2, 0], [0, 0, 5, 2], [0, 0, 0, 3], [0, 0, 0, 0]]
    out = max_flow_min_cut(C, 0, 3)
    assert out["flow"] == 5.0
    assert out["cut_capacity"] == 5.0
    assert out["min_cut_source_side"] == [0]
    assert sorted(out["cut_edges"]) == [(0, 1), (0, 2)]


def test_the_cut_capacity_is_recomputed_from_the_original_capacities():
    rng = random.Random(9)
    n = 5
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.6:
                C[i][j] = rng.randint(1, 10)
    out = max_flow_min_cut(C, 0, n - 1)
    S = set(out["min_cut_source_side"])
    manual = sum(C[i][j] for i in S for j in range(n) if j not in S)
    assert manual == pytest.approx(out["cut_capacity"])


def test_a_disconnected_sink_has_zero_flow():
    C = [[0, 5, 0], [0, 0, 0], [0, 0, 0]]
    out = max_flow_min_cut(C, 0, 2)
    assert out["flow"] == 0.0
    assert out["cut_capacity"] == 0.0


def test_flow_validation():
    with pytest.raises(ValueError, match="must be square"):
        max_flow_min_cut([[0, 1], [0]])
    with pytest.raises(ValueError, match="non-negative"):
        max_flow_min_cut([[0, -1], [0, 0]])
    with pytest.raises(ValueError, match="must differ"):
        max_flow_min_cut([[0, 1], [0, 0]], 0, 0)
