"""netstat against igraph 2.3 (closeness, betweenness, eigen_centrality, page_rank, transitivity,
assortativity_degree, global_efficiency, diameter, radius, modularity, edge/vertex connectivity).

Reference values printed by R for the graph below; information centrality,
vulnerability and redundancy are checked against their definitions.
tests/cross/test-morie_vs_igraph.R repeats the comparison live.
"""

import pytest

from morie.fn.netstat import (
    centralities,
    modularity_score,
    network_connectivity,
    network_summary,
    node_vulnerability,
    shortest_path_lengths,
)

N = 14
EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (0, 10),
    (1, 2),
    (1, 4),
    (1, 7),
    (2, 3),
    (2, 7),
    (2, 10),
    (2, 12),
    (3, 4),
    (3, 7),
    (3, 8),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (4, 5),
    (4, 11),
    (4, 12),
    (5, 6),
    (6, 7),
    (6, 8),
    (6, 13),
    (7, 8),
    (8, 9),
    (8, 12),
    (8, 13),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
]
IG = {
    "closeness": [
        0.040000000000000001,
        0.041666666666666664,
        0.047619047619047616,
        0.055555555555555552,
        0.047619047619047616,
        0.037037037037037035,
        0.040000000000000001,
        0.047619047619047616,
        0.047619047619047616,
        0.037037037037037035,
        0.043478260869565216,
        0.043478260869565216,
        0.050000000000000003,
        0.041666666666666664,
    ],
    "betweenness": [
        2.1499999999999995,
        1.9083333333333334,
        5.7607142857142843,
        11.686904761904762,
        11.402380952380954,
        1.4999999999999998,
        4.6761904761904756,
        6.4452380952380954,
        8.6095238095238109,
        0.82499999999999996,
        6.3833333333333337,
        1.2,
        5.0857142857142854,
        1.3666666666666667,
    ],
    "eigenvector": [
        0.4728574066597655,
        0.48095696462384069,
        0.7544424403056289,
        0.99999999999999989,
        0.67358612654648709,
        0.20272074358516201,
        0.39325342162047877,
        0.63020138219082344,
        0.68784920447291864,
        0.24081720468726928,
        0.57947703602831624,
        0.58144269421456196,
        0.80684106927929566,
        0.54876677037698962,
    ],
    "pagerank": [
        0.06178761856974354,
        0.061525775173256754,
        0.086741856571813963,
        0.11227679928504421,
        0.08977890287800859,
        0.03724681912583612,
        0.065006378841564794,
        0.074555958377503978,
        0.089899414094299468,
        0.036442034086727373,
        0.076423517896564083,
        0.060581925384968621,
        0.086312144445975403,
        0.061420855268693134,
    ],
    "transitivity": 0.40714285714285714,
    "local": [
        0.5,
        0.5,
        0.40000000000000002,
        0.39285714285714285,
        0.26666666666666666,
        0,
        0.33333333333333331,
        0.40000000000000002,
        0.40000000000000002,
        0,
        0.29999999999999999,
        0.66666666666666663,
        0.46666666666666667,
        0.66666666666666663,
    ],
    "assortativity": -0.039689034369883934,
    "efficiency": 0.66117216117216104,
    "diameter": 3,
    "radius": 2,
    "modularity": 0.014692378328741929,
    "edge_conn": 2,
    "vertex_conn": 2,
}


def test_centralities_equal_igraph():
    c = centralities(N, EDGES)
    for k in ("closeness", "betweenness", "eigenvector", "pagerank"):
        assert c[k] == pytest.approx(IG[k], abs=1e-12)


def test_summary_modularity_connectivity_equal_igraph():
    s = network_summary(N, EDGES)
    assert s.transitivity == pytest.approx(IG["transitivity"], abs=1e-15)
    assert [v if v == v else -1 for v in s.local_transitivity] == pytest.approx(IG["local"], abs=1e-15)
    assert s.assortativity == pytest.approx(IG["assortativity"], abs=1e-13)
    assert s.efficiency == pytest.approx(IG["efficiency"], abs=1e-15)
    assert (s.diameter, s.radius) == (IG["diameter"], IG["radius"])
    memb = [0 if i < 7 else 1 for i in range(N)]
    assert modularity_score(N, EDGES, memb) == pytest.approx(IG["modularity"], abs=1e-15)
    c = network_connectivity(N, EDGES)
    assert (c.edge, c.vertex) == (IG["edge_conn"], IG["vertex_conn"])


def test_small_graphs_by_hand():
    path = [(0, 1), (1, 2), (2, 3)]
    c = centralities(4, path)
    assert c.degree == [1, 2, 2, 1] and c.betweenness == [0.0, 2.0, 2.0, 0.0]
    assert c.closeness == pytest.approx([1 / 6, 1 / 4, 1 / 4, 1 / 6])
    # information centrality equals n / sum of effective resistances on a tree
    assert centralities(3, [(0, 1), (1, 2)]).information == pytest.approx([1.0, 1.5, 1.0], abs=1e-12)
    s = network_summary(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    assert s.transitivity == pytest.approx(3 / 5) and s.density == pytest.approx(4 / 6)
    d = network_summary(3, [(0, 1), (1, 0), (1, 2)], directed=True)
    assert d.reciprocity == pytest.approx(2 / 3)
    assert shortest_path_lengths(3, [(0, 1, 2.5), (1, 2, 1.0), (0, 2, 5.0)], weighted=True)[0][2] == 3.5
    v = node_vulnerability(4, path)
    # efficiency of the path: (1 + 1/2 + 1/3 + 1 + 1/2 + 1) * 2 / 12; without node 0: (1 + 1/2 + 1) * 2 / 12
    E, E0 = (1 + 0.5 + 1 / 3 + 1 + 0.5 + 1) / 6, (1 + 0.5 + 1) / 6
    assert v.vulnerability[0] == pytest.approx((E - E0) / E, abs=1e-15)
    tri = node_vulnerability(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    assert tri.redundancy[2] == pytest.approx(2 / 3) and tri.effective_size[2] == pytest.approx(3 - 2 / 3)
    ring = network_connectivity(5, [(i, (i + 1) % 5) for i in range(5)])
    assert (ring.edge, ring.vertex) == (2, 2)
    full = network_connectivity(4, [(i, j) for i in range(4) for j in range(i + 1, 4)])
    assert full.vertex == 3
