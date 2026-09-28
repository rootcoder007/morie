import math

from morie.fn.clusterdensity import denclue, flame_clustering, growing_neural_gas, possibilistic_fcm


def _blobs():
    pts = []
    for c, (cx, cy) in enumerate([(0, 0), (6, 1), (2, 7)]):
        for i in range(15):
            a = (i * 2.399963 + c) % (2 * math.pi)
            r = 0.4 + 1.3 * ((i * 0.618034 + c * 0.3) % 1)
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return pts


TRUTH = [1] * 15 + [2] * 15 + [3] * 15


def test_denclue_attractors_are_mean_shift_fixed_points():
    X, h = _blobs(), 0.8
    r = denclue(X, h, 0.005)
    assert r.cluster == TRUTH
    c = (2 * math.pi * h * h) ** -1 / len(X)
    for y, f in zip(r.attractors[:5], r.density[:5]):
        w = [math.exp(-((y[0] - q[0]) ** 2 + (y[1] - q[1]) ** 2) / (2 * h * h)) for q in X]
        m = [sum(wi * q[t] for wi, q in zip(w, X)) / sum(w) for t in range(2)]
        assert math.dist(m, y) < 1e-3
        assert abs(f - c * sum(w)) < 1e-12
    assert denclue([[0.0], [0.3], [0.1], [9.0], [9.2], [30.0]], 0.5, 0.2).cluster == [1, 1, 1, 2, 2, 0]


def test_flame_supports_and_memberships():
    X = _blobs()
    r = flame_clustering(X, knn=6)
    assert r.cluster == TRUTH
    for row in r.membership:
        assert abs(sum(row) - 1) < 1e-9
    kmax = min(int(math.sqrt(45)) + 10, 44)

    def dens(i):
        d = sorted(math.dist(X[i], X[j]) for j in range(45) if j != i)[:kmax]
        k = 6
        while k < kmax and d[k] == d[5]:
            k += 1
        return 1 / (sum(d[:k]) + 1e-9), k

    for s in r.supports:
        di, k = dens(s - 1)
        nb = sorted((j for j in range(45) if j != s - 1), key=lambda j: (math.dist(X[s - 1], X[j]), j))[:k]
        assert all(di >= dens(j)[0] for j in nb)


def test_pfcm_fixed_point():
    X = _blobs()
    r = possibilistic_fcm(X, [[0, 0], [5, 0], [0, 5]])
    assert r.cluster == TRUTH
    for c, v in enumerate(r.centers):
        w = []
        for x in X:
            d2 = [math.dist(x, u) ** 2 for u in r.centers]
            u = 1 / sum(d2[c] / d2[j] for j in range(3))
            t = 1 / (1 + d2[c] / r.gamma[c])
            w.append(u * u + t * t)
        m = [sum(wi * x[q] for wi, x in zip(w, X)) / sum(w) for q in range(2)]
        assert math.dist(m, v) < 1e-7


def test_gng_graph():
    X = _blobs()
    r = growing_neural_gas(X, n_signals=4000, max_nodes=12, seed=3)
    assert r.cluster == TRUTH and len(r.nodes) <= 12
    assert all(1 <= a < b <= len(r.nodes) for a, b in r.edges)
