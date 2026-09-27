"""Spatial weight operators.

Checked against spdep 1.4 (poly2nb queen/rook on irregular polygons, card,
diffnb, nb2blocknb) and spatialreg 1.3 (eigenw bounds, invIrW) to 3e-15;
tests/cross/test-morie_vs_spdep.R repeats that in R.
"""

import pytest

from morie.fn.swops import (
    block_weights,
    compare_neighbours,
    error_operator,
    lag_operator,
    neighbour_cardinality,
    polygon_contiguity,
    regime_weights,
    rho_bounds,
)

W3 = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]


def test_rho_bounds_path_and_rejects_general_w():
    r = rho_bounds(W3)
    # eigenvalues of the row-standardised 3-path are -1, 0, 1
    assert r.eigenvalues == pytest.approx([-1.0, 0.0, 1.0], abs=1e-12)
    assert (round(r.lower, 6), round(r.upper, 6)) == (-1.0, 1.0)
    assert r.spectral_radius == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        rho_bounds([[0, 0.3], [0.9, 0]])


def test_error_operator_inverts_i_minus_rho_w():
    M = error_operator([[0, 1], [1, 0]], 0.5)
    assert [[round(v, 6) for v in r] for r in M] == [[1.333333, 0.666667], [0.666667, 1.333333]]
    A = [[1.0, -0.5], [-0.5, 1.0]]
    prod = [[sum(A[i][k] * M[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert [v for r in prod for v in r] == pytest.approx([1.0, 0.0, 0.0, 1.0], abs=1e-12)


def test_lag_operator_powers():
    assert lag_operator(W3, [1.0, 2.0, 4.0]) == [2.0, 2.5, 2.0]
    assert lag_operator(W3, [1.0, 2.0, 4.0], power=2) == [2.5, 2.0, 2.5]
    assert lag_operator(W3, [1.0, 2.0, 4.0], power=0) == [1.0, 2.0, 4.0]


def test_block_and_regime_weights():
    assert block_weights(["a", "b", "a"]) == [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    full = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert regime_weights(full, [1, 1, 2]) == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert regime_weights(full, [1, 1, 1], row_standardize=True)[0] == [0.0, 0.5, 0.5]


def test_polygon_contiguity_queen_and_rook():
    def sq(x, y):
        return [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]

    polys = [sq(0, 0), sq(1, 0), sq(1, 1)]
    assert polygon_contiguity(polys, queen=False) == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    # squares 0 and 2 touch at the single corner (1, 1)
    assert polygon_contiguity(polys, queen=True)[0] == [0.0, 1.0, 1.0]


def test_cardinality_and_comparison():
    path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    r = neighbour_cardinality(path)
    assert (r.cardinality, r.table, r.n_links, r.islands) == ([1, 2, 1], {1: 2, 2: 1}, 4, [])
    c = compare_neighbours(path, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert (c.difference, c.only_first, c.only_second, c.identical) == ([[2], [], [0]], 0, 2, False)
