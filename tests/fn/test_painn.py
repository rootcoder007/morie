"""Tests for painn.painn (PaiNN update block, Schutt et al. 2021, eq. 9-11)."""

import math

import pytest

from morie.fn.painn import painn

S = [0.3, -1.2]
V = [[0.5, -0.2], [1.0, 0.4], [-0.3, 0.8]]          # 3 spatial dims x F = 2
U = [[0.7, -0.1], [0.2, 1.3]]
W = [[1.1, 0.5], [-0.4, 0.9]]


def _phi(s, dot, nrm):
    # any scalar network works; it only ever sees invariants
    return {"ds": [s[f] + dot[f] + 0.5 * nrm[f] for f in range(2)],
            "gate": [math.tanh(s[f] - dot[f]) for f in range(2)]}


def _rot(v, th):
    c, s = math.cos(th), math.sin(th)
    R = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return [[sum(R[a][b] * v[b][f] for b in range(3)) for f in range(2)] for a in range(3)]


def test_painn_basic():
    """U v and V v act on the feature axis; the scalar input is
    <U v, V v> per feature, and dv = gate * (U v), recomputed."""
    r = painn(S, V, U, W, _phi)
    assert isinstance(r, dict)
    Uv = [[sum(U[f][g] * V[a][g] for g in range(2)) for f in range(2)] for a in range(3)]
    Wv = [[sum(W[f][g] * V[a][g] for g in range(2)) for f in range(2)] for a in range(3)]
    dot = [sum(Uv[a][f] * Wv[a][f] for a in range(3)) for f in range(2)]
    assert r["scalar_from_vectors"] == pytest.approx(dot, rel=1e-14)
    gate = [math.tanh(S[f] - dot[f]) for f in range(2)]
    for a in range(3):
        assert r["dv"][a] == pytest.approx([gate[f] * Uv[a][f] for f in range(2)], rel=1e-14)


def test_painn_edge():
    """Rotating every vector leaves the scalar update unchanged and
    rotates the vector update: invariance and equivariance."""
    th = 0.7
    a, b = painn(S, V, U, W, _phi), painn(S, _rot(V, th), U, W, _phi)
    assert b["ds"] == pytest.approx(a["ds"], rel=1e-13)
    rot = _rot(a["dv"], th)
    for i in range(3):
        assert b["dv"][i] == pytest.approx(rot[i], rel=1e-13, abs=1e-15)
