"""Tests for spfunc: functional PCA, graph layers, persistent homology, possibilistic clustering, REML."""

import math

from morie.fn.spfunc import (
    crossed_random_effects,
    curve_fpca,
    gat_layer,
    gcn_layer,
    nested_random_effects,
    persistence_landscape,
    possibilistic_cmeans,
    reml_components,
    rips_persistence,
    sgc_ridge,
)


def test_fpca_on_uniform_grid_matches_scaled_pca():
    t = [i / 4 for i in range(5)]
    C = [[math.sin(3 * v * (1 + 0.2 * i)) + 0.3 * i * v for v in t] for i in range(8)]
    r = curve_fpca(C, t, 2)
    # eigenfunctions are orthonormal under the trapezoid inner product
    w = [0.125, 0.25, 0.25, 0.25, 0.125]
    for a in range(2):
        for b in range(2):
            ip = sum(w[j] * r.eigenfunctions[a][j] * r.eigenfunctions[b][j] for j in range(5))
            assert abs(ip - (a == b)) <= 1e-12
    # score variance equals the eigenvalue
    for k in range(2):
        s = [row[k] for row in r.scores]
        assert abs(sum(v * v for v in s) / 7 - r.eigenvalues[k]) <= 1e-12


def test_gcn_and_sgc_and_gat():
    A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    H = [[1.0], [2.0], [4.0]]
    d = [2, 3, 2]
    S = [[(A[i][j] + (i == j)) / math.sqrt(d[i] * d[j]) for j in range(3)] for i in range(3)]
    out = gcn_layer(A, H, [[0.5]], activation="linear")
    for i in range(3):
        assert abs(out[i][0] - 0.5 * sum(S[i][j] * H[j][0] for j in range(3))) <= 1e-15
    r = sgc_ridge(A, H, [1.0, 2.0, 3.5], k=1, l2=0.0)
    assert abs(sum(r.fitted) - 6.5) <= 1e-12
    g = gat_layer(A, H, [[1.0]], [0.0, 0.0])
    assert abs(g.output[1][0] - 7.0 / 3.0) <= 1e-15
    assert abs(sum(g.attention[1]) - 1.0) <= 1e-15


def test_rips_persistence_square_and_clusters():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    d = rips_persistence(sq).diagram
    assert [r for r in d if r[0] == 0] == [[0, 0.0, 1.0]] * 3 + [[0, 0.0, math.inf]]
    assert [r for r in d if r[0] == 1] == [[1, 1.0, math.sqrt(2.0)]]
    two = rips_persistence([(0.0, 0.0), (0.5, 0.0), (5.0, 0.0), (5.5, 0.0)]).diagram
    assert [0, 0.0, 4.5] in two


def test_landscape_tents():
    L = persistence_landscape([[0, 0.0, 2.0], [0, 1.0, 3.0], [1, 0.5, 0.7]], [0.5, 1.5, 2.5], k_max=2, dimension=0)
    assert L == [[0.5, 0.5, 0.5], [0.0, 0.5, 0.0]]


def test_pcm_fixed_point():
    X = [
        [math.sin(i * 1.3) + (3 if i % 3 == 0 else 0), math.cos(i * 0.7) + (2 if i % 3 == 1 else 0)] for i in range(24)
    ]
    r = possibilistic_cmeans(X, [[0.0, 0.0], [3.0, 0.5], [0.5, 2.0]], omega=[1.0, 1.5, 0.8])
    for j in range(3):
        t = [1 / (1 + sum((a - b) ** 2 for a, b in zip(x, r.centers[j])) / r.omega[j]) for x in X]
        v = [sum(t[i] ** 2 * X[i][c] for i in range(24)) / sum(q * q for q in t) for c in range(2)]
        assert max(abs(a - b) for a, b in zip(v, r.centers[j])) <= 1e-8


def test_reml_one_way_balanced_is_anova():
    y = [1.0, 1.2, 3.0, 3.3, 2.0, 2.4]
    Z = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    r = reml_components(y, [[1.0]] * 6, [Z])
    msw = (0.02 + 0.045 + 0.08) / 3
    means = [1.1, 3.15, 2.2]
    msb = 2 * sum((m - 2.15) ** 2 for m in means) / 2
    assert abs(r.variances[1] - msw) <= 1e-10 and abs(r.variances[0] - (msb - msw) / 2) <= 1e-10
    j = list(range(24))
    y2 = [1 + 0.5 * math.sin(i * 1.7) + (i % 4) * 0.3 + (i % 3) * 0.2 for i in j]
    assert len(crossed_random_effects(y2, [i % 4 for i in j], [i % 3 for i in j]).variances) == 3
    nest = nested_random_effects(y2, [i // 8 for i in j], [(i // 2) % 4 for i in j])
    assert nest.variances[0] <= 1e-9 and abs(nest.variances[2] - sum((v - sum(y2) / 24) ** 2 for v in y2) / 23) <= 1e-9
