"""Tests for grclus (Karypis & Kumar 1998, METIS)."""

from morie.fn.grclus import (balance_bisection, coarsen, edge_cut,
                             graph_clustering, kernighan_lin,
                             match_vertices, metis_partition,
                             total_edge_weight, _as_graph)


def _grid(a, b):
    n = a * b
    A = [[0.0] * n for _ in range(n)]
    for i in range(a):
        for j in range(b):
            u = i * b + j
            if i + 1 < a:
                A[u][(i + 1) * b + j] = A[(i + 1) * b + j][u] = 1.0
            if j + 1 < b:
                A[u][i * b + j + 1] = A[i * b + j + 1][u] = 1.0
    return A


def _path(n):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        A[i][i + 1] = A[i + 1][i] = 1.0
    return A


def _cliques(count, m):
    n = count * m
    A = [[0.0] * n for _ in range(n)]
    for g in range(count):
        b = g * m
        for i in range(b, b + m):
            for j in range(i + 1, b + m):
                A[i][j] = A[j][i] = 1.0
    for g in range(count if count > 2 else 1):
        b, nb = g * m, ((g + 1) % count) * m
        A[b][nb] = A[nb][b] = 1.0
    return A


def test_coarsening_identities():
    adj = _as_graph(_grid(6, 6))
    vw = [1.0] * 36
    w_e = total_edge_weight(adj)
    for scheme in ("hem", "rm", "lem"):
        mate = match_vertices(adj, scheme, seed=5)
        assert all(mate[mate[u]] == u for u in range(36))
        assert all(mate[u] == u or mate[u] in adj[u] for u in range(36))
        w_m = sum(adj[u][mate[u]] for u in range(36)
                  if mate[u] != u) / 2.0
        adj2, vw2, mapping = coarsen(adj, vw, mate)
        # eq. 1
        assert abs(total_edge_weight(adj2) - (w_e - w_m)) < 1e-12
        assert abs(sum(vw2) - sum(vw)) < 1e-12
        coarse = [i % 2 for i in range(len(adj2))]
        fine = [coarse[mapping[u]] for u in range(36)]
        assert abs(edge_cut(adj2, coarse) - edge_cut(adj, fine)) < 1e-12


def test_known_optimal_cuts():
    assert metis_partition(_cliques(2, 20), 2)["edge_cut"] == 1.0
    assert metis_partition(_cliques(4, 12), 4)["edge_cut"] == 4.0
    assert metis_partition(_path(40), 2)["edge_cut"] == 1.0
    cuts = [metis_partition(_grid(8, 8), 2, seed=s)["edge_cut"]
            for s in range(1, 9)]
    assert min(cuts) == 8.0            # the grid optimum
    assert max(cuts) <= 12.0           # and it never wanders far


def test_balance_is_respected():
    for sd in range(1, 9):
        res = metis_partition(_grid(8, 8), 2, seed=sd)
        assert max(res["part_weights"]) <= 32.0 + 0.03 * 64.0 + 1e-9
    for k in (2, 4, 8):
        res = metis_partition(_grid(8, 8), k, seed=4)
        assert res["sizes"] == [64 // k] * k


def test_vertex_weights_are_what_is_balanced():
    w = [1.0] * 32 + [3.0] * 32
    res = metis_partition(_grid(8, 8), 2, weights=w, seed=4)
    assert abs(res["part_weights"][0] -
               res["part_weights"][1]) <= 0.06 * sum(w)


def test_balance_bisection_pulls_a_lopsided_split_back():
    adj = _as_graph(_grid(6, 6))
    bad = [0] * 30 + [1] * 6
    fixed = balance_bisection(adj, [1.0] * 36, bad, 18.0, 0.03)
    assert abs(sum(1 for t in fixed if t == 0) - 18) <= 2


def test_kl_refinement_never_increases_the_cut():
    adj = _as_graph(_grid(8, 8))
    bad = [0] * 32 + [1] * 32
    before = edge_cut(adj, bad)
    refined, after = kernighan_lin(adj, [1.0] * 64, bad, 32.0)
    assert after <= before + 1e-12
    assert abs(after - edge_cut(adj, refined)) < 1e-9


def test_edge_weights_change_the_answer():
    A = _cliques(2, 12)
    A[0][12] = A[12][0] = 50.0
    for u, v in ((3, 15), (4, 16)):
        A[u][v] = A[v][u] = 1.0
    res = metis_partition(A, 2)
    assert res["partition"][0] == res["partition"][12]   # heavy edge kept
    assert res["edge_cut"] < 0.5 * edge_cut(A, [0] * 12 + [1] * 12)


def test_all_routes_run():
    for mt in ("hem", "rm", "lem"):
        for ini in ("gggp", "ggp"):
            for ref in ("bkl", "kl"):
                res = metis_partition(_grid(6, 6), 2, matching=mt,
                                      initial=ini, refinement=ref, seed=4)
                assert res["edge_cut"] <= 10.0


def test_trivial_k():
    assert metis_partition(_grid(4, 4), 1)["edge_cut"] == 0.0
    res = metis_partition(_grid(4, 4), 16)
    assert sorted(res["sizes"]) == [1] * 16
    assert res["edge_cut"] == total_edge_weight(_as_graph(_grid(4, 4)))


def test_validation():
    for call in (lambda: metis_partition([[0.0, 1.0], [1.0, 0.0]], 0),
                 lambda: metis_partition([[0.0, 1.0], [1.0, 0.0]], 5),
                 lambda: metis_partition([[0.0, 1.0, 0.0],
                                          [1.0, 0.0, 0.0]], 2),
                 lambda: metis_partition([[0.0, 1.0], [2.0, 0.0]], 2),
                 lambda: metis_partition([[0.0, -1.0], [-1.0, 0.0]], 2),
                 lambda: metis_partition(_grid(4, 4), 2,
                                         matching="greedy"),
                 lambda: metis_partition(_grid(4, 4), 2,
                                         initial="spectral"),
                 lambda: metis_partition(_grid(4, 4), 2, refinement="fm"),
                 lambda: metis_partition(_grid(4, 4), 2, tolerance=1.5),
                 lambda: metis_partition(_grid(4, 4), 2,
                                         weights=[1.0] * 15),
                 lambda: metis_partition(_grid(4, 4), 2,
                                         weights=[0.0] * 16),
                 lambda: edge_cut(_grid(4, 4), [0, 1])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    a = graph_clustering(_grid(4, 4), 2, seed=4)
    b = metis_partition(_grid(4, 4), 2, seed=4)
    assert a["edge_cut"] == b["edge_cut"]
