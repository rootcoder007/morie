"""Tests for swinmw (Swin v1 Eq 4 window MSA + relative bias)."""

import math

from morie.fn.swinmw import swin_msa_window, swinmw

X = [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 1.0]]]


def test_swinmw_bias_index_anchor():
    # Liu et al. (2021): B values are looked up from the (2M-1)^2 table
    # by relative position. For M = 2, token p=(0,0) vs q=(1,1) has
    # (di,dj) = (-1,-1) -> table[0][0]; p=q -> table[1][1].
    T = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
    r = swinmw(X, 2, relative_bias=T)
    B = r["bias"]
    assert B[0][3] == 10.0   # (0,0) vs (1,1): di=dj=-1
    assert B[0][0] == 50.0   # same token: di=dj=0
    assert B[3][0] == 90.0   # (1,1) vs (0,0): di=dj=+1
    assert B[1][2] == 30.0   # (0,1) vs (1,0): di=-1, dj=+1


def test_swinmw_reductions():
    # M = 1, zero bias, identity projections: single-token windows can
    # only attend to themselves -> output equals the input map.
    r = swinmw(X, 1)
    assert r["n_windows"] == 4 and r["tokens_per_window"] == 1
    for i in range(2):
        for j in range(2):
            assert r["output"][i][j] == [float(v) for v in X[i][j]]
    # M = 2, zero bias: equals directly-coded global attention over the
    # 4 tokens (one window covers the whole map).
    r2 = swinmw(X, 2)
    toks = [X[0][0], X[0][1], X[1][0], X[1][1]]
    d = 2
    for p in range(4):
        s = [sum(a * b for a, b in zip(toks[p], t)) / math.sqrt(d) for t in toks]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        w = [v / z for v in e]
        want = [sum(wi * t[c] for wi, t in zip(w, toks)) for c in range(d)]
        i, j = divmod(p, 2)
        got = r2["output"][i][j]
        assert all(abs(a - b) < 1e-12 for a, b in zip(got, want))
    # huge diagonal bias pins each token to itself even at M = 2
    T = [[0.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 0.0]]
    r3 = swinmw(X, 2, relative_bias=T)
    for i in range(2):
        for j in range(2):
            assert all(abs(a - b) < 1e-9 for a, b in zip(r3["output"][i][j], X[i][j]))
    assert swin_msa_window is swinmw
