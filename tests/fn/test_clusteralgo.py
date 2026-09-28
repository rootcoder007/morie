import itertools
import math

from morie.fn.clusops import cut_tree, hierarchical_clustering
from morie.fn.clusteralgo import affinity_propagation, chameleon, clarans, cure_clustering, diana


def _blobs():
    pts = []
    for c, (cx, cy) in enumerate([(0, 0), (6, 1), (2, 7)]):
        for i in range(15):
            a = (i * 2.399963 + c) % (2 * math.pi)
            r = 0.4 + 1.3 * ((i * 0.618034 + c * 0.3) % 1)
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return pts


def _same_partition(a, b):
    return all((a[i] == a[j]) == (b[i] == b[j]) for i in range(len(a)) for j in range(len(a)))


def test_affinity_propagation_fixed_assignment():
    X = _blobs()
    r = affinity_propagation(X)
    ex = [e - 1 for e in r.exemplars]
    S = [[-(math.dist(a, b) ** 2) for b in X] for a in X]
    for i in range(len(X)):
        best = max(range(len(ex)), key=lambda j: (S[i][ex[j]], -j))
        assert r.cluster[i] == (ex.index(i) + 1 if i in ex else best + 1)
    for j, e in enumerate(ex):  # exemplar maximises summed similarity within its cluster
        mem = [i for i in range(len(X)) if r.cluster[i] == j + 1]
        assert all(sum(S[i][e] for i in mem) >= sum(S[i][q] for i in mem) - 1e-12 for q in mem)
    assert _same_partition(r.cluster, [1] * 15 + [2] * 15 + [3] * 15)


def test_clarans_reaches_exhaustive_optimum():
    X = [[0, 0], [1, 0.5], [0.3, 1.2], [7, 7], [8, 6.5], [7.5, 8.2], [3.5, 3.9], [4.2, 3.1]]
    D = [[math.dist(a, b) for b in X] for a in X]
    best = min(sum(min(D[i][m] for m in med) for i in range(8)) for med in itertools.combinations(range(8), 2))
    r = clarans(X, 2, numlocal=5, seed=2)
    assert abs(r.cost - best) < 1e-12
    assert abs(r.cost - sum(min(D[i][m - 1] for m in r.medoids) for i in range(8))) < 1e-12


def test_cure_single_linkage_limit():
    X = _blobs()[:20]
    D = [[math.dist(a, b) for b in X] for a in X]
    h = hierarchical_clustering(D, "single")
    for k in (2, 3, 4):
        assert _same_partition(cure_clustering(X, k, n_rep=20, alpha=0.0).cluster, cut_tree(h.merge, k))


def test_diana_dc_and_split():
    D = [[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]]
    r = diana(D, 2)
    assert r.cluster == [1, 1, 2, 2] and abs(r.dc - (1 - 1 / 6)) < 1e-15
    X = _blobs()
    Dm = [[math.dist(a, b) for b in X] for a in X]
    r = diana(Dm, 3)
    assert _same_partition(r.cluster, [1] * 15 + [2] * 15 + [3] * 15)
    full = max(max(row) for row in Dm)
    last = [0.0] * 45
    for s in sorted(r.splits, key=lambda s: -len(s["members"])):
        for i in s["members"]:
            last[i] = s["diameter"]
    assert abs(r.dc - sum(1 - v / full for v in last) / 45) < 1e-12


def test_chameleon_blobs():
    assert _same_partition(chameleon(_blobs(), 3, n_neighbors=4).cluster, [1] * 15 + [2] * 15 + [3] * 15)
