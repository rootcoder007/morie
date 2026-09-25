"""Tests for egnnL: E(n)-equivariant graph networks.

Satorras, Hoogeboom and Welling (2021), eqs. (3)-(6). The defining
property is checked directly: transform the input coordinates by a
rotation or reflection Q and a translation g, and the output
coordinates must transform the same way while the features stay put.
"""

import math

import pytest

from morie.fn.egnnL import (coord_update, edge_message, egcl, egnn_layer,
                            equivariance_error, run_egnn)


def phi_e(hi, hj, d2, a):
    return [math.tanh(hi[0] - hj[0] + 0.3 * d2), math.sin(hi[1] * hj[0] + d2)]


def phi_x(m):
    return 0.2 * math.tanh(m[0] - m[1])


def phi_h(h, m):
    return [h[0] + 0.5 * math.tanh(m[0]), h[1] - 0.3 * m[1]]


H = [[0.1, 1.0], [0.7, -0.4], [-0.3, 0.2], [0.5, 0.5]]
X = [[0.0, 0.0, 0.0], [1.0, 0.2, -0.5], [0.3, 1.1, 0.4], [-0.8, 0.5, 0.9]]


def _rot(a, b, c):
    """A proper rotation from three Euler angles."""
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    Rz = [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]
    Ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    Rx = [[1, 0, 0], [0, cc, -sc], [0, sc, cc]]
    mm = lambda A, B: [[sum(A[i][k] * B[k][j] for k in range(3))
                        for j in range(3)] for i in range(3)]
    return mm(mm(Rz, Ry), Rx)


def _apply(Q, g, P):
    return [[sum(Q[a][b] * p[b] for b in range(3)) + g[a] for a in range(3)]
            for p in P]


def test_egnnL_basic():
    """Eq. (3): messages see positions only through ||x_i - x_j||^2."""
    m = edge_message(H[0], H[1], X[0], X[1], phi_e)
    d2 = sum((X[0][k] - X[1][k]) ** 2 for k in range(3))
    assert m == phi_e(H[0], H[1], d2, None)
    Q, g = _rot(0.4, -1.1, 2.0), [3.0, -2.0, 0.5]
    Xt = _apply(Q, g, X)
    assert edge_message(H[0], H[1], Xt[0], Xt[1], phi_e) == pytest.approx(m, abs=1e-12)


def test_coordinate_update_is_equation_4():
    M = [[None if i == j else phi_e(H[i], H[j], 0.0, None) for j in range(4)]
         for i in range(4)]
    out = coord_update(X, M, phi_x)
    C = 1.0 / 3.0
    for i in range(4):
        want = [X[i][d] + C * sum((X[i][d] - X[j][d]) * phi_x(M[i][j])
                                  for j in range(4) if j != i) for d in range(3)]
        assert out[i] == pytest.approx(want, rel=1e-14)


def test_rotation_reflection_and_translation_equivariance():
    base = run_egnn(H, X, 3, phi_e, phi_x, phi_h)
    reflect = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    for Q in (_rot(0.4, -1.1, 2.0), reflect, _rot(2.5, 0.3, -0.7)):
        g = [3.0, -2.0, 0.5]
        moved = run_egnn(H, _apply(Q, g, X), 3, phi_e, phi_x, phi_h)
        want = _apply(Q, g, base["X"])
        for i in range(4):
            assert moved["X"][i] == pytest.approx(want[i], abs=1e-10)
            assert moved["H"][i] == pytest.approx(base["H"][i], abs=1e-10)
    err = equivariance_error(H, X, phi_e, phi_x, phi_h, _rot(1, 2, 3), [1, 1, 1])
    assert err["equivariant"] and err["invariant"]
    assert egnn_layer is run_egnn


def test_momentum_variant_is_equivariant_too():
    V = [[0.1, 0.0, -0.2], [0.0, 0.3, 0.1], [-0.1, 0.1, 0.0], [0.2, -0.2, 0.2]]
    phi_v = lambda h: 1.0 + 0.1 * h[0]
    base = egcl(H, X, phi_e, phi_x, phi_h, V=V, mode="momentum", phi_v=phi_v)
    Q, g = _rot(0.9, 0.1, -1.4), [0.0, 5.0, -1.0]
    Vt = [[sum(Q[a][b] * v[b] for b in range(3)) for a in range(3)] for v in V]
    moved = egcl(H, _apply(Q, g, X), phi_e, phi_x, phi_h, V=Vt,
                 mode="momentum", phi_v=phi_v)
    want = _apply(Q, g, base["X"])
    for i in range(4):
        assert moved["X"][i] == pytest.approx(want[i], abs=1e-10)


def test_egnnL_edge():
    with pytest.raises(ValueError, match="at least 2 particles"):
        coord_update([[0.0, 0.0]], [[None]], phi_x)
    with pytest.raises(ValueError, match="mode must be"):
        egcl(H, X, phi_e, phi_x, phi_h, mode="spin")
    with pytest.raises(ValueError, match="needs V and phi_v"):
        egcl(H, X, phi_e, phi_x, phi_h, mode="momentum")
