"""Tests for rmsdtr (Kabsch optimal-superposition RMSD).

Replaces the generated stub, which imported ``rmsd``.
"""

import math

from morie.fn.rmsdtr import rmsdtr


def _shape():
    return [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]


def _rotate_z(P, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [[c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]] for p in P]


def test_a_shape_matches_itself_exactly():
    P = _shape()
    res = rmsdtr(P, P)
    assert res["estimate"] < 1e-12


def test_a_pure_rotation_and_translation_is_removed():
    P = _shape()
    Q = [[v + 5.0 for v in p] for p in _rotate_z(P, 0.7)]
    res = rmsdtr(P, Q)
    assert res["estimate"] < 1e-9
    assert abs(res["det"] - 1.0) < 1e-9      # a proper rotation


def test_the_rotation_matrix_is_orthonormal():
    P = _shape()
    Q = _rotate_z(P, 1.1)
    R = rmsdtr(P, Q)["rotation"]
    for i in range(3):
        assert abs(sum(R[i][k] ** 2 for k in range(3)) - 1.0) < 1e-9
        for j in range(i + 1, 3):
            assert abs(sum(R[i][k] * R[j][k]
                           for k in range(3))) < 1e-9


def test_a_real_deformation_leaves_a_positive_rmsd():
    P = _shape()
    Q = [list(p) for p in P]
    Q[0][0] += 1.0
    res = rmsdtr(P, Q)
    assert res["estimate"] > 0.1


def test_weights_can_ignore_a_displaced_atom():
    P = _shape()
    Q = [list(p) for p in P]
    Q[4] = [10.0, 10.0, 10.0]
    heavy = rmsdtr(P, Q)["estimate"]
    ignored = rmsdtr(P, Q, weights=[1.0, 1.0, 1.0, 1.0, 0.0])["estimate"]
    assert ignored < heavy
    assert ignored < 1e-9


def test_a_reflection_is_not_used():
    # a mirrored copy cannot be matched by a proper rotation, so the
    # determinant stays +1 and the RMSD stays positive
    P = _shape()
    Q = [[-p[0], p[1], p[2]] for p in P]
    res = rmsdtr(P, Q)
    assert abs(res["det"] - 1.0) < 1e-9
    assert res["estimate"] > 1e-6


def test_validation():
    P = _shape()
    for call in (lambda: rmsdtr(P[:2], P[:2]),
                 lambda: rmsdtr(P, [p[:2] for p in P]),
                 lambda: rmsdtr(P, P, weights=[0.0] * 5),
                 lambda: rmsdtr(P, P, weights=[-1.0] * 5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
