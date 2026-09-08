"""Tests for moetop (GShard top-k routing + Switch aux loss)."""

import math

from morie.fn.moetop import moe_topk_routing, moetop

E0 = [[1.0, 0.0], [0.0, 1.0]]
E1 = [[2.0, 0.0], [0.0, 2.0]]
E2 = [[0.0, 1.0], [1.0, 0.0]]


def test_moetop_uniform_anchor_switch():
    # Switch Eqs 4-6: with symmetric tokens the routing is exactly
    # balanced, f = P = (1/2, 1/2), so aux = alpha * N * (2 * 1/4)
    # = alpha. The paper states alpha = 1e-2.
    X = [[1.0, 0.0], [0.0, 1.0]]
    Wg = [[1.0, 0.0], [0.0, 1.0]]
    r = moetop(X, Wg, [E0, E1], k=1, alpha=0.01)
    assert r["f"] == [0.5, 0.5]
    assert all(abs(p - 0.5) < 1e-15 for p in r["P"])
    assert abs(r["aux_loss"] - 0.01) < 1e-15


def test_moetop_k1_and_dense_reduction():
    X = [[1.0, 0.2]]
    Wg = [[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]]
    # k = 1: the output IS the argmax expert applied to x
    r = moetop(X, Wg, [E0, E1, E2], k=1)
    i = r["topk_indices"][0][0]
    ex = [E0, E1, E2][i]
    want = [sum(X[0][a] * ex[a][c] for a in range(2)) for c in range(2)]
    assert all(abs(u - v) < 1e-12 for u, v in zip(r["output"][0], want))
    assert r["topk_gates"][0] == [1.0]
    # k = N: gates renormalise to the full softmax -> dense mixture
    r3 = moetop(X, Wg, [E0, E1, E2], k=3)
    g = r3["gates"][0]
    assert abs(sum(g) - 1.0) < 1e-12
    dense = [sum(g[i] * sum(X[0][a] * [E0, E1, E2][i][a][c] for a in range(2))
                 for i in range(3)) for c in range(2)]
    assert all(abs(u - v) < 1e-12 for u, v in zip(r3["output"][0], dense))


def test_moetop_wrapper():
    X = [[1.0, 0.0]]
    Wg = [[1.0, 0.0], [0.0, 1.0]]
    a = moe_topk_routing(x=X, W_g=Wg, experts=[E0, E1], k=2)
    b = moetop(X, Wg, [E0, E1], k=2)
    assert a["output"] == b["output"]
