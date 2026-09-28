"""swbuild: constructions checked against hand-built neighbour sets (spdep and GWmodel checked in tests/cross)."""

import math

import pytest

from morie.fn.swbuild import (
    distance_band_weights,
    grid_contiguity,
    inverse_distance_weights,
    kernel_weights,
    knn_weights,
    symmetrize_weights,
    weights_components,
)

P = [(0.0, 0.0), (1.0, 0.0), (2.5, 0.0), (0.0, 1.2), (5.0, 5.0)]


def test_grid_contiguity():
    q = grid_contiguity(3, 3)
    assert sum(q[4]) == 8 and sum(q[0]) == 3 and q[0][4] == 1.0
    r = grid_contiguity(3, 3, type="rook")
    assert sum(r[4]) == 4 and r[0][4] == 0.0 and r[0] == [0, 1, 0, 1, 0, 0, 0, 0, 0]
    t = grid_contiguity(3, 3, type="rook", torus=True)
    assert all(sum(row) == 4 for row in t) and t[0][2] == 1.0 and t[0][6] == 1.0
    assert all(q[i][j] == q[j][i] for i in range(9) for j in range(9))
    with pytest.raises(ValueError):
        grid_contiguity(2, 2, type="bishop")


def test_distance_knn_inverse_kernel():
    D = distance_band_weights(P, 1.25, d1=1.0)
    assert D[0] == [0.0, 0.0, 0.0, 1.0, 0.0]  # d(0,1) = 1 is excluded by d1 < d, d(0,3) = 1.2 included
    K = knn_weights(P, 2)
    # point 4 at (5, 5): nearest are (2.5, 0) at 5.59 and (0, 1.2) at 6.28
    assert K[0] == [0.0, 1.0, 0.0, 1.0, 0.0] and K[4] == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert sum(map(sum, K)) == 10
    Iw = inverse_distance_weights(P, power=2.0, d2=2.0, row_standardize=True)
    raw = [0.0, 1.0, 0.0, 1 / 1.44, 0.0]
    assert Iw[0] == pytest.approx([v / sum(raw) for v in raw])
    g = kernel_weights(P, 1.5, kernel="gaussian")
    assert g[0][2] == pytest.approx(math.exp(-0.5 * (2.5 / 1.5) ** 2))
    a = kernel_weights(P, 3, kernel="bisquare", adaptive=True)
    b = sorted(math.sqrt((P[0][0] - q[0]) ** 2 + (P[0][1] - q[1]) ** 2) for q in P)[
        2
    ]  # the third nearest counting the point itself
    assert b == 1.2 and a[0][1] == pytest.approx((1 - (1 / 1.2) ** 2) ** 2) and a[0][3] == 0.0
    assert kernel_weights(P, 2.0, kernel="tricube", diagonal=False)[1][1] == 0.0
    with pytest.raises(ValueError):
        kernel_weights(P, 9, adaptive=True)


def test_symmetrize_components():
    A = knn_weights(P, 1)
    U = symmetrize_weights(A)
    assert all(U[i][j] == U[j][i] for i in range(5) for j in range(5))
    assert all(U[i][j] >= A[i][j] for i in range(5) for j in range(5))
    X = symmetrize_weights(A, method="intersection")
    assert all(X[i][j] <= A[i][j] for i in range(5) for j in range(5))
    assert symmetrize_weights([[0, 2], [0, 0]], method="average") == [[0.0, 1.0], [1.0, 0.0]]
    c = weights_components(distance_band_weights(P, 1.6))
    # links 0-1 (1.0), 0-3 (1.2), 1-2 (1.5), 1-3 (1.56); point 4 is isolated
    assert (c.n_components, c.component, c.sizes, c.connected) == (2, [1, 1, 1, 1, 2], [4, 1], False)
