# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Symmetrically normalised graph convolution layer.

Kipf and Welling (2017), "Semi-supervised classification with graph
convolutional networks", ICLR 2017, arXiv:1609.02907, equation (2):

    H' = sigma(D^{-1/2} A D^{-1/2} H W),

with D = diag(row sums of A).  This is the layer BEFORE the
renormalisation trick of equation (8): the adjacency has no self-loop
added, so an isolated node receives nothing.  Isolated nodes have
D_ii = 0 and their normalising factor is taken to be zero rather than
infinite.  sigma is the rectifier.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gcn"]


def _norm_adj(M, n, self_loops):
    B = [[M[i][j] + (1.0 if (self_loops and i == j) else 0.0) for j in range(n)] for i in range(n)]
    d = [sum(B[i]) for i in range(n)]
    s = [0.0 if d[i] <= 0 else d[i] ** -0.5 for i in range(n)]
    return [[s[i] * B[i][j] * s[j] for j in range(n)] for i in range(n)]


def gcn(G, X, W):
    """One propagation step of the un-renormalised GCN layer."""
    M = core.mat(G)
    n = len(M)
    if n == 0:
        raise ValueError("gcn: graph is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("gcn: adjacency matrix must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("gcn: X must have one row per node")
    Wm = core.mat(W)
    if len(Wm) != len(H[0]):
        raise ValueError("gcn: W must have one row per input feature")
    An = _norm_adj(M, n, False)
    AH = core.matmul(An, H)
    Z = core.matmul(AH, Wm)
    Hout = [[core.relu(v) for v in row] for row in Z]
    flat = [v for row in Hout for v in row]
    return RichResult(
        title="Graph convolutional layer",
        summary_lines=[("nodes", n), ("out_features", len(Hout[0]))],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": Hout,
            "preactivation": Z,
            "n": n,
            "method": "H' = relu(D^{-1/2} A D^{-1/2} X W), Kipf & Welling (2017) eq. (2)",
        },
    )
