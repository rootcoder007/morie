# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Graph attention layer.

Velickovic, Cucurull, Casanova, Romero, Lio and Bengio (2018), "Graph
attention networks", ICLR 2018, arXiv:1710.10903, equations (1)-(4):

    e_ij   = LeakyReLU( a' [ W h_i || W h_j ] ),
    alpha  = softmax_j over the neighbourhood N_i (which includes i),
    h_i'   = sigma( sum_{j in N_i} alpha_ij W h_j ),

and equation (5) averages the heads in the final layer.  With no
parameters supplied the deterministic choice W = I and a = 1 is used,
so e_ij = LeakyReLU(sum(h_i) + sum(h_j)); the negative slope is the
paper's 0.2.  Every head then sees the same input and averaging them
is exact, which is what makes the heads argument testable.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["graph_attention_net"]

SLOPE = 0.2


def _lrelu(z):
    return z if z > 0 else SLOPE * z


def graph_attention_net(G, X, heads=1):
    """One multi-head GAT layer with identity weights."""
    M = core.mat(G)
    n = len(M)
    if n == 0:
        raise ValueError("graph_attention_net: graph is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("graph_attention_net: adjacency matrix must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("graph_attention_net: X must have one row per node")
    nh = int(heads)
    if nh < 1:
        raise ValueError("graph_attention_net: heads must be at least 1")
    p = len(H[0])
    rs = [sum(H[i]) for i in range(n)]
    out = [[0.0] * p for _ in range(n)]
    alphas = []
    for i in range(n):
        nb = [j for j in range(n) if j == i or M[i][j] != 0.0]
        e = [_lrelu(rs[i] + rs[j]) for j in nb]
        mx = max(e)
        w = [math.exp(v - mx) for v in e]
        tot = sum(w)
        w = [v / tot for v in w]
        alphas.append(w)
        for a, j in enumerate(nb):
            for c in range(p):
                out[i][c] += w[a] * H[j][c]
    out = [[core.sigmoid(v) for v in row] for row in out]
    flat = [v for row in out for v in row]
    return RichResult(
        title="Graph attention network layer",
        summary_lines=[("nodes", n), ("heads", nh)],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": out,
            "alpha_first": alphas[0],
            "heads": nh,
            "n": n,
            "method": "GAT eqs. (1)-(4) with W = I, a = 1, LeakyReLU slope 0.2",
        },
    )
