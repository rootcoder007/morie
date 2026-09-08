"""Tests for sctraj (Street et al. 2018, Slingshot)."""

import math

from morie.fn.sctraj import (average_curve, cluster_distances, cosine_cdf,
                             lineages_from_tree, minimum_spanning_tree,
                             principal_curve, pseudotime_trajectory,
                             sctraj, shrinkage_weight)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _spearman(a, b):
    n = len(a)
    pa, pb = [0] * n, [0] * n
    for k, i in enumerate(sorted(range(n), key=lambda i: a[i])):
        pa[i] = k
    for k, i in enumerate(sorted(range(n), key=lambda i: b[i])):
        pb[i] = k
    m = (n - 1) / 2.0
    num = sum((pa[i] - m) * (pb[i] - m) for i in range(n))
    da = math.sqrt(sum((pa[i] - m) ** 2 for i in range(n)))
    db = math.sqrt(sum((pb[i] - m) ** 2 for i in range(n)))
    return num / (da * db) if da and db else 0.0


X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
     [10.0, 0.0], [10.0, 4.0], [10.0, -4.0], [10.0, 8.0]]
LAB = ["a"] * 4 + ["b"] * 4


def test_equation_1_by_hand():
    info = cluster_distances(X, LAB, cov="full")
    ca, cb = info["centers"]["a"], info["centers"]["b"]
    Sa, Sb = info["covariances"]["a"], info["covariances"]["b"]
    d = [ca[0] - cb[0], ca[1] - cb[1]]
    P = [[Sa[i][j] + Sb[i][j] for j in range(2)] for i in range(2)]
    det = P[0][0] * P[1][1] - P[0][1] * P[1][0]
    inv = [[P[1][1] / det, -P[0][1] / det], [-P[1][0] / det, P[0][0] / det]]
    want = math.sqrt(sum(d[i] * inv[i][j] * d[j]
                         for i in range(2) for j in range(2)))
    assert abs(info["distances"][("a", "b")] - want) < 1e-9
    assert abs(info["distances"][("a", "b")] -
               info["distances"][("b", "a")]) < 1e-12


def test_the_euclidean_route_is_the_plain_distance():
    e = cluster_distances(X, LAB, cov="euclidean")
    assert abs(e["distances"][("a", "b")] - 9.617692030835672) < 1e-9
    f = cluster_distances(X, LAB, cov="full")
    assert abs(e["distances"][("a", "b")] -
               f["distances"][("a", "b")]) > 1.0


def _blobs():
    centres = {"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [2.0, 0.0],
               "D": [3.0, 1.0], "E": [3.0, -1.0]}
    pts, labs = [], []
    r = _lcg(11)
    for name, c in centres.items():
        for _ in range(12):
            pts.append([c[0] + 0.08 * _gauss(r), c[1] + 0.08 * _gauss(r)])
            labs.append(name)
    return pts, labs


def test_the_mst_and_its_lineages():
    pts, labs = _blobs()
    d = cluster_distances(pts, labs, cov="diagonal")
    tree = minimum_spanning_tree(d["distances"], d["clusters"])
    edges = set(frozenset((a, b)) for a, b, _ in tree["edges"])
    assert len(tree["edges"]) == 4
    assert edges == {frozenset(("A", "B")), frozenset(("B", "C")),
                     frozenset(("C", "D")), frozenset(("C", "E"))}
    lins = lineages_from_tree(tree, "A")
    assert len(lins) == 2
    assert all(p[:3] == ["A", "B", "C"] for p in lins)
    assert sorted(p[-1] for p in lins) == ["D", "E"]
    assert len(lineages_from_tree(tree, "C")) == 3


def test_terminal_state_supervision():
    pts, labs = _blobs()
    d = cluster_distances(pts, labs, cov="diagonal")
    sup = minimum_spanning_tree(d["distances"], d["clusters"],
                                ends=["D", "E"])
    edges = set(frozenset((a, b)) for a, b, _ in sup["edges"])
    assert frozenset(("C", "D")) in edges
    assert frozenset(("C", "E")) in edges
    assert len(sup["edges"]) == 4


def test_pseudotime_recovers_the_generating_parameter():
    r = _lcg(7)
    T = [i / 79.0 for i in range(80)]
    curve = [[3.0 * t, 2.0 * t * t] for t in T]
    noisy = [[p[0] + 0.05 * _gauss(r), p[1] + 0.05 * _gauss(r)]
             for p in curve]
    fit = principal_curve(noisy, [curve[0], curve[-1]])
    assert abs(_spearman(fit["pseudotime"], T)) > 0.99
    assert abs(min(fit["pseudotime"])) < 1e-12
    assert max(fit["distance"]) < 0.5


def test_equation_4_properties():
    assert abs(shrinkage_weight(0.0, 2.0, 5.0) - 1.0) < 1e-12
    assert abs(shrinkage_weight(6.0, 2.0, 5.0)) < 1e-12
    assert abs(shrinkage_weight(3.5, 2.0, 5.0) - 0.5) < 1e-12
    ts = [2.0 + 3.0 * k / 60.0 for k in range(61)]
    ws = [shrinkage_weight(t, 2.0, 5.0) for t in ts]
    assert all(ws[k] >= ws[k + 1] - 1e-12 for k in range(len(ws) - 1))
    assert abs(ws[0] - 1.0) < 1e-12 and abs(ws[-1]) < 1e-12


def test_the_printed_form_of_equation_4_is_discontinuous():
    lit = shrinkage_weight(2.0, 2.0, 5.0, arg="as_printed")
    assert abs(lit - 1.0) > 0.1


def test_the_cosine_kernel_cdf():
    assert abs(cosine_cdf(-0.5)) < 1e-15
    assert abs(cosine_cdf(0.5) - 1.0) < 1e-15
    assert abs(cosine_cdf(0.0) - 0.5) < 1e-15
    g = [-0.5 + k / 100.0 for k in range(101)]
    assert all(cosine_cdf(g[k]) <= cosine_cdf(g[k + 1]) + 1e-15
               for k in range(len(g) - 1))


def _branching():
    r = _lcg(2024)
    spec = [("T0", 0.0, 0.0), ("T1", 1.2, 0.0), ("T2", 2.4, 0.0),
            ("U0", 3.8, 1.6), ("U1", 5.0, 3.2),
            ("L0", 3.8, -1.6), ("L1", 5.0, -3.2)]
    cells, clab = [], []
    for name, cx, cy in spec:
        for _ in range(18):
            cells.append([cx + 0.18 * _gauss(r), cy + 0.18 * _gauss(r)])
            clab.append(name)
    return cells, clab


def test_branching_gives_two_lineages_sharing_a_trunk():
    cells, clab = _branching()
    res = sctraj(cells, clab, root="T0", cov="diagonal")
    assert res["n_lineages"] == 2
    assert all(p[0] == "T0" for p in res["lineages"])
    assert all(p[:3] == ["T0", "T1", "T2"] for p in res["lineages"])
    trunk = [i for i in range(len(cells)) if clab[i].startswith("T")]
    rho = abs(_spearman([res["pseudotime"][0][i] for i in trunk],
                        [cells[i][0] for i in trunk]))
    assert rho > 0.95


def test_shrinkage_makes_the_curves_share_a_start_point():
    cells, clab = _branching()
    shrunk = sctraj(cells, clab, root="T0", cov="diagonal")
    plain = sctraj(cells, clab, root="T0", cov="diagonal", shrink=False)
    assert max(abs(shrunk["curves"][0][0][j] - shrunk["curves"][1][0][j])
               for j in range(2)) < 1e-9
    assert max(abs(plain["curves"][0][0][j] - plain["curves"][1][0][j])
               for j in range(2)) > 1e-9


def test_cell_weights_separate_the_branches():
    cells, clab = _branching()
    res = sctraj(cells, clab, root="T0", cov="diagonal")
    up = [i for i in range(len(cells)) if clab[i].startswith("U")]
    lo = [i for i in range(len(cells)) if clab[i].startswith("L")]
    best_u = set(max(range(2), key=lambda k: res["weights"][k][i])
                 for i in up)
    best_l = set(max(range(2), key=lambda k: res["weights"][k][i])
                 for i in lo)
    assert len(best_u) == 1 and len(best_l) == 1
    assert best_u != best_l
    assert all(0.0 <= v <= 1.0 for col in res["weights"] for v in col)


def test_the_average_curve_is_the_mean():
    c1 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    c2 = [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]
    avg = average_curve([c1, c2], n_points=5)
    assert all(abs(p[1] - 1.0) < 1e-9 for p in avg)
    assert all(abs(p[1]) < 1e-9 for p in average_curve([c1], n_points=5))


def test_validation():
    pts, labs = _blobs()
    d = cluster_distances(pts, labs, cov="diagonal")
    tree = minimum_spanning_tree(d["distances"], d["clusters"])
    cells, clab = _branching()
    for call in (lambda: cluster_distances([], []),
                 lambda: cluster_distances(X, LAB[:-1]),
                 lambda: cluster_distances(X, LAB, cov="mahal"),
                 lambda: cluster_distances(X, ["a"] * 8),
                 lambda: cluster_distances(X, LAB, weights=[1.0] * 7),
                 lambda: minimum_spanning_tree(d["distances"], ["A"]),
                 lambda: minimum_spanning_tree(d["distances"],
                                               d["clusters"], ends=["Z"]),
                 lambda: minimum_spanning_tree(d["distances"],
                                               d["clusters"],
                                               ends=list(d["clusters"])),
                 lambda: lineages_from_tree(tree, "Z"),
                 lambda: principal_curve(pts, [[0.0, 0.0]]),
                 lambda: principal_curve(pts, [[0.0, 0.0], [1.0, 1.0]],
                                         max_iter=0),
                 lambda: average_curve([]),
                 lambda: average_curve([[[0.0, 0.0]]], n_points=1),
                 lambda: shrinkage_weight(1.0, 0.0, 2.0, arg="literal"),
                 lambda: sctraj(cells, clab[:-1], root="T0"),
                 lambda: sctraj([[1.0], [float("nan")]], ["a", "b"],
                                root="a")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert pseudotime_trajectory is sctraj
