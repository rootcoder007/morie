"""Optimisers and community detection: DE and SA on Philox, Leiden (Traag 2019), Girvan-Newman; igraph references."""

import math

from morie.fn._rng import random_uniform
from morie.fn.comgir import girvan_newman
from morie.fn.deopti import differential_evolution
from morie.fn.sa_opt import simulated_annealing
from morie.fn.scleid import leiden_clustering

KARATE = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
]


def karate():
    A = [[0.0] * 34 for _ in range(34)]
    for i, j in KARATE:
        A[i][j] = A[j][i] = 1.0
    return A


def sbm():
    n = 60
    U = random_uniform(n * n, seed=11, stream=0)
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            u = float(U[i * n + j])
            if u < (0.35 if i // 15 == j // 15 else 0.04):
                S[i][j] = S[j][i] = round(0.5 + u * 3, 6)
    return S


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def modularity(A, lab):
    n = len(A)
    m2 = sum(sum(r) for r in A)
    k = [sum(r) for r in A]
    return sum(A[i][j] - k[i] * k[j] / m2 for i in range(n) for j in range(n) if lab[i] == lab[j]) / m2


def test_differential_evolution_solves_rosenbrock_and_is_pinned():
    pop = [[-2 + 0.2 * i, 2 - 0.15 * i] for i in range(20)]
    r = differential_evolution(rosen, pop, generations=300)
    assert r["estimate"] < 1e-20 and max(abs(v - 1) for v in r["x"]) < 1e-10
    r = differential_evolution(rosen, pop, generations=3, seed=4)
    assert r["estimate"] == 0.3692960000000009 and r["x"] == [0.4400000000000001, 0.16999999999999987]
    assert r["evals"] == 20 + 20 * 3 and min(r["fvals"]) == r["estimate"]
    assert all(abs(rosen(p) - f) < 1e-15 for p, f in zip(r["population"], r["fvals"]))


def test_simulated_annealing_uses_philox_and_is_pinned():
    r = simulated_annealing(rosen, [-1.2, 1.0], step=0.1, n_iter=2000, seed=3)
    assert (
        r["fun"] == 3.890860177211241e-05
        and r["x"] == [0.9942293634111342, 0.9887288467288058]
        and r["n_accepted"] == 87
    )
    assert abs(rosen(r["x"]) - r["fun"]) < 1e-15 and r["fun"] <= min(r["trace"]) + 1e-15


def test_leiden_reaches_the_karate_optimum_igraph():
    A = karate()
    for seed in (0, 1, 2):
        r = leiden_clustering(A, seed=seed)
        # igraph::cluster_leiden(objective_function = "modularity") best: 0.4197896
        assert abs(r["estimate"] - 0.41978961209730437) < 1e-12
        assert abs(modularity(A, r["labels"]) - r["estimate"]) < 1e-12
        assert r["connected"] and r["n_communities"] == 4
    assert r["labels"] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        2,
        2,
        1,
        0,
        0,
        0,
        2,
        2,
        1,
        0,
        2,
        0,
        2,
        0,
        2,
        3,
        3,
        3,
        2,
        3,
        3,
        2,
        2,
        3,
        2,
        2,
    ]
    # CPM on karate at gamma = 0.1: igraph best H (paper eq. 2) = 43.1
    assert abs(leiden_clustering(A, resolution=0.1, quality="cpm", seed=1)["estimate"] - 43.1) < 1e-12


def test_leiden_random_refinement_is_pinned():
    S = sbm()
    for seed, lab, q in zip(
        (0, 1, 2),
        [
            [
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                2,
                3,
                2,
                4,
                3,
                5,
                3,
                5,
                3,
                3,
                3,
                5,
                5,
                3,
                3,
                6,
                7,
                7,
                6,
                8,
                7,
                9,
                8,
                8,
                7,
                7,
                6,
                7,
                6,
                6,
                10,
                11,
                12,
                12,
                13,
                11,
                12,
                10,
                11,
                10,
                12,
                12,
                10,
                10,
                11,
            ],
            [
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                2,
                2,
                3,
                4,
                2,
                5,
                2,
                6,
                6,
                2,
                6,
                6,
                6,
                2,
                2,
                7,
                8,
                8,
                7,
                9,
                8,
                10,
                9,
                9,
                8,
                8,
                7,
                8,
                7,
                7,
                11,
                12,
                13,
                13,
                14,
                12,
                13,
                14,
                12,
                11,
                14,
                12,
                11,
                11,
                12,
            ],
            [
                0,
                0,
                1,
                0,
                2,
                3,
                0,
                2,
                3,
                0,
                3,
                3,
                2,
                0,
                2,
                1,
                4,
                1,
                5,
                4,
                6,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                7,
                7,
                7,
                7,
                8,
                7,
                8,
                7,
                8,
                7,
                8,
                7,
                7,
                8,
                9,
                10,
                11,
                12,
                12,
                13,
                11,
                12,
                13,
                11,
                10,
                13,
                11,
                10,
                10,
                11,
            ],
        ],
        [50.900972, 51.069916000000006, 50.532725000000006],
    ):
        r = leiden_clustering(S, resolution=0.3, quality="cpm", seed=seed)
        assert r["labels"] == lab and abs(r["estimate"] - q) < 1e-12 and r["connected"]


def test_girvan_newman_matches_igraph_edge_betweenness():
    A = karate()
    r = girvan_newman(A)
    # igraph::cluster_edge_betweenness: max modularity 0.4012985, 5 communities
    assert abs(r["estimate"] - 0.4012984878369494) < 1e-12
    assert abs(modularity(A, r["labels"]) - r["estimate"]) < 1e-12
    assert r["labels"] == [
        0,
        0,
        1,
        0,
        2,
        2,
        2,
        0,
        3,
        4,
        2,
        0,
        0,
        0,
        3,
        3,
        2,
        0,
        3,
        0,
        3,
        0,
        3,
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        3,
        1,
        3,
        3,
    ]
    assert r["removed"][0] == (0, 31)  # largest edge betweenness 71.39286 in igraph
    two = girvan_newman(A, n_communities=2)
    assert two["n_communities"] == 2
    A6 = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    r = girvan_newman(A6)
    assert r["labels"] == [0, 0, 0, 1, 1, 1] and r["removed"][0] == (2, 3) and abs(r["estimate"] - 5 / 14) < 1e-12


def test_edge_betweenness_values_match_igraph():
    from morie.fn.comgir import _edge_betweenness

    A = karate()
    eb = _edge_betweenness([[j for j in range(34) if A[i][j]] for i in range(34)], 34)
    assert abs(eb[(0, 31)] - 71.39285714285714) < 1e-9  # igraph::edge_betweenness
    assert abs(sum(eb.values()) - 1351.0) < 1e-9  # igraph total


def _refine_brute(adj, n, r, comm, theta, us_of):
    # Traag et al. (2019) MergeNodesSubset with every cut weight recomputed from scratch
    ref = list(range(n))
    for c in sorted(set(comm)):
        S = [v for v in range(n) if comm[v] == c]
        us = us_of(2 * len(S))

        def cut(C_members):
            return sum(a for x in C_members for y, a in adj[x].items() if y in S and y not in C_members)

        order = sorted(range(len(S)), key=lambda i: (us[i], i))
        for t, i in enumerate(order):
            v = S[i]
            if sum(1 for u in S if ref[u] == ref[v]) > 1 or cut([v]) < r * (len(S) - 1) - 1e-12:
                continue
            cand, gains = [ref[v]], [0.0]
            for C in sorted({ref[u] for u in adj[v] if u in S and u != v}):
                mem = [u for u in S if ref[u] == C]
                if cut(mem) < r * len(mem) * (len(S) - len(mem)) - 1e-12:
                    continue
                g = sum(adj[v].get(u, 0.0) for u in mem) - r * len(mem)
                if g >= 0.0:
                    cand.append(C)
                    gains.append(g)
            gm = max(gains)
            ps = [math.exp((g - gm) / theta) for g in gains]
            x, acc, pick = us[len(S) + t] * sum(ps), 0.0, cand[-1]
            for C, q in zip(cand, ps):
                acc += q
                if x < acc:
                    pick = C
                    break
            ref[v] = pick
    return ref


def test_leiden_refinement_matches_brute_force_merge_nodes_subset():
    from morie.fn.scleid import _Draws, _fast_move, _refine

    for gs, n, p in ((6, 18, 0.3), (65, 21, 0.2), (77, 19, 0.4), (3, 15, 0.5)):
        U = random_uniform(n * n, seed=gs, stream=0)
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if float(U[i * n + j]) < p:
                    A[i][j] = A[j][i] = 1.0
        adj = [{j: A[i][j] for j in range(n) if A[i][j]} for i in range(n)]
        for gamma in (0.2, 0.4, 0.6):
            for seed in (0, 1, 2):
                d1, d2 = _Draws(seed), _Draws(seed)
                comm = _fast_move(adj, [1.0] * n, gamma, list(range(n)), d1)
                assert _fast_move(adj, [1.0] * n, gamma, list(range(n)), d2) == comm
                assert _refine(adj, [1.0] * n, gamma, comm, 0.01, d1) == _refine_brute(
                    adj, n, gamma, comm, 0.01, d2.take
                )
