"""Tests for scumap (McInnes, Healy & Melville 2018, UMAP)."""

import math

from morie.fn.scumap import (fit_ab, fuzzy_simplicial_set, scumap,
                             smooth_knn_dist, spectral_layout,
                             umap_singlecell)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _blobs(n_per=15, n_clust=3, dim=8, seed=7, sep=10.0):
    r = _lcg(seed)
    X, lab = [], []
    for c in range(n_clust):
        for _ in range(n_per):
            X.append([(sep * c if k == c else 0.0) + _gauss(r)
                      for k in range(dim)])
            lab.append(c)
    return X, lab


def test_smooth_knn_dist_solves_algorithm_3():
    for dists in ([0.5, 0.7, 0.9, 1.3, 2.0, 2.6, 3.1, 4.0],
                  [0.1, 5.0, 5.1, 5.2, 5.3]):
        n = len(dists)
        sigma, rho = smooth_knn_dist(dists, n)
        total = sum(math.exp(-max(0.0, v - rho) / sigma) for v in dists)
        assert abs(total - math.log(n, 2)) < 1e-4
        assert rho == min(dists)


def test_equidistant_neighbours_have_no_solution():
    # every term is exp(0) = 1 for any sigma, so the sum is stuck at n
    tied = [1.0, 1.0, 1.0, 1.0]
    sigma, rho = smooth_knn_dist(tied, 4)
    total = sum(math.exp(-max(0.0, v - rho) / sigma) for v in tied)
    assert abs(total - 4.0) < 1e-12
    assert 0.0 < sigma < float("inf")


def test_a_sparser_neighbourhood_gets_a_larger_sigma():
    tight = smooth_knn_dist([1.0, 1.1, 1.2, 1.3], 4)[0]
    wide = smooth_knn_dist([1.0, 3.0, 5.0, 7.0], 4)[0]
    assert wide > tight


def test_the_membership_formula_and_the_t_conorm():
    X, _ = _blobs()
    g = fuzzy_simplicial_set(X, n_neighbors=6)
    A, B = g["A"], g["B"]
    n = len(X)
    for i in (0, 20, n - 1):
        assert abs(max(A[i]) - 1.0) < 1e-9        # nearest neighbour
        assert sum(1 for v in A[i] if v > 0) == 6
        for j in range(n):
            want = A[i][j] + A[j][i] - A[i][j] * A[j][i]
            assert abs(B[i][j] - want) < 1e-15
            assert abs(B[i][j] - B[j][i]) < 1e-15
            assert -1e-15 <= B[i][j] <= 1.0 + 1e-12
    i, j = 0, g["neighbours"][0][2]
    d = math.sqrt(sum((X[i][k] - X[j][k]) ** 2 for k in range(8)))
    assert abs(A[i][j] -
               math.exp(-max(0.0, d - g["rho"][i]) / g["sigma"][i])) < 1e-12


def test_ab_matches_the_reference_defaults():
    a, b = fit_ab(0.1, 1.0)
    assert abs(a - 1.577) < 0.02
    assert abs(b - 0.895) < 0.02
    assert fit_ab(1.0, 1.0)[0] < a


def test_the_spectral_layout_and_the_papers_misprint():
    X, _ = _blobs()
    B = fuzzy_simplicial_set(X, n_neighbors=6)["B"]
    Y = spectral_layout(B, 2)
    assert len(Y) == len(X) and len(Y[0]) == 2
    printed = spectral_layout(B, 2, laplacian="as_printed")
    # the paper's literal D^{1/2}(D-A)D^{1/2} is a different operator
    assert max(abs(Y[i][c] - printed[i][c])
               for i in range(len(X)) for c in range(2)) > 1e-6


def test_planted_clusters_separate_in_the_embedding():
    X, lab = _blobs()
    Y = umap_singlecell(X, n_neighbors=6, n_epochs=120,
                        seed=1)["embedding"]
    n = len(X)

    def centroid(c):
        pts = [Y[i] for i in range(n) if lab[i] == c]
        return [sum(p[k] for p in pts) / len(pts) for k in range(2)]

    cs = [centroid(c) for c in range(3)]
    within = []
    for c in range(3):
        pts = [Y[i] for i in range(n) if lab[i] == c]
        within.append(sum(math.dist(p, cs[c]) for p in pts) / len(pts))
    between = [math.dist(cs[i], cs[j])
               for i in range(3) for j in range(i + 1, 3)]
    assert min(between) / (sum(within) / 3.0) > 3.0


def test_near_neighbours_survive_the_projection():
    X, _ = _blobs()
    n = len(X)
    Y = umap_singlecell(X, n_neighbors=6, n_epochs=120,
                        seed=1)["embedding"]

    def knn(points, i, k, dim):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: sum((points[i][c] - points[j][c]) ** 2
                                         for c in range(dim)))
        return set(order[:k])

    kept = sum(len(knn(X, i, 5, 8) & knn(Y, i, 5, 2)) for i in range(n))
    assert kept / float(5 * n) > 0.4


def test_seed_reproducibility_and_both_inits():
    X, _ = _blobs()
    a = umap_singlecell(X, n_neighbors=6, n_epochs=20, seed=3)
    b = umap_singlecell(X, n_neighbors=6, n_epochs=20, seed=3)
    assert a["embedding"] == b["embedding"]
    r = umap_singlecell(X, n_neighbors=6, n_epochs=20, seed=3,
                        init="random")
    assert r["init"] == "random"
    assert len(r["embedding"]) == len(X)


def test_validation():
    X, _ = _blobs(n_per=6)
    for call in (lambda: umap_singlecell(X, n_neighbors=1),
                 lambda: umap_singlecell(X, n_neighbors=len(X) + 5),
                 lambda: umap_singlecell(X, init="pca"),
                 lambda: umap_singlecell(X, n_components=0),
                 lambda: umap_singlecell(X, learning_rate=0.0),
                 lambda: umap_singlecell(X, n_epochs=0),
                 lambda: umap_singlecell([[1.0], [float("nan")],
                                          [3.0]]),
                 lambda: umap_singlecell([[1.0, 2.0], [3.0]]),
                 lambda: fit_ab(-1.0),
                 lambda: fit_ab(0.1, spread=0.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert scumap is umap_singlecell
