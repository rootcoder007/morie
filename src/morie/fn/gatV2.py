# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""GATv2: the attention nonlinearity moved inside.

Brody, Alon and Yahav (2022), "How attentive are graph attention
networks?", ICLR 2022, arXiv:2105.14491, equation (7):

    e(h_i, h_j) = a' LeakyReLU( W [ h_i || h_j ] )

against GAT's e = LeakyReLU(a' W[h_i || h_j]) of equation (3).  Only
the order of the linear layer and the nonlinearity changes, and that
is exactly what turns static attention into dynamic attention.  With
W = I and a = 1 the score becomes
sum_c LeakyReLU(h_ic) + sum_c LeakyReLU(h_jc), whereas GAT's stays
LeakyReLU(sum h_i + sum h_j); the two differ whenever a feature vector
mixes signs, which is the discriminating case exercised in the tests.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gat_v2"]

SLOPE = 0.2


def _lrelu(z):
    return z if z > 0 else SLOPE * z


def gat_v2(A, X):
    """One GATv2 layer with identity weights."""
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("gat_v2: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("gat_v2: adjacency matrix must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("gat_v2: X must have one row per node")
    p = len(H[0])
    g = [sum(_lrelu(v) for v in H[i]) for i in range(n)]
    out = [[0.0] * p for _ in range(n)]
    alphas = []
    for i in range(n):
        nb = [j for j in range(n) if j == i or M[i][j] != 0.0]
        e = [g[i] + g[j] for j in nb]
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
        title="GATv2 layer",
        summary_lines=[("nodes", n)],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": out,
            "alpha_first": alphas[0],
            "n": n,
            "method": "GATv2 eq. (7): a' LeakyReLU(W[h_i || h_j]) with W = I, a = 1",
        },
    )


def cheatsheet():
    return "gatV2: GATv2 attention layer"


# compact alias per ledger/NAMING.md
gatv2 = gat_v2
