"""Tests for linatt (Wang et al. 2020 Linformer, Eq 7)."""

import math

from morie.fn.linatt import linatt, linformer_linear_attention

Q = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
K = [[1.0, 0.5], [0.2, 1.0], [0.0, 0.3]]
V = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def _direct(Q, K, V):
    d = len(Q[0])
    out = []
    for q in Q:
        s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d) for k in K]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        w = [v / z for v in e]
        out.append([sum(wi * vr[c] for wi, vr in zip(w, V))
                    for c in range(len(V[0]))])
    return out


def test_linatt_identity_projection_anchor():
    # Eq 7 with k = n and E = F = I reduces EXACTLY to standard scaled
    # dot-product attention (independently coded here).
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    r = linatt(Q, K, V, I3, I3)
    want = _direct(Q, K, V)
    for a, b in zip(r["output"], want):
        for x, y in zip(a, b):
            assert abs(x - y) < 1e-12


def test_linatt_projection_shapes_and_pooling():
    # k = 1: E and F average the keys/values; every query sees a single
    # projected key, so the attention weight is exactly 1 and the
    # output is the F-projected value row.
    E = [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
    F = [[0.2, 0.3, 0.5]]
    r = linatt(Q, K, V, E, F)
    assert r["k"] == 1
    fv = [0.2 * V[0][c] + 0.3 * V[1][c] + 0.5 * V[2][c] for c in (0, 1)]
    for row, w in zip(r["output"], r["weights"]):
        assert abs(w[0] - 1.0) < 1e-15
        assert all(abs(x - y) < 1e-12 for x, y in zip(row, fv))
    assert linformer_linear_attention is linatt
