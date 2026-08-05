# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Katz centrality with an explicit baseline weight beta.

Katz (1953), Psychometrika 18(1):39-43, doi:10.1007/BF02289026, in the
scaled form

    C_K = (I - alpha A)^{-1} beta 1,

so beta = 1 recovers the plain resolvent and beta scales every score
linearly.  The graph is supplied as its adjacency matrix.
"""

from __future__ import annotations

from . import _array_core as np
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["katz_centrality"]


def katz_centrality(G, alpha=0.1, beta=1.0):
    """Katz scores of adjacency matrix G with baseline beta."""
    M = core.mat(G)
    n = len(M)
    if n == 0:
        raise ValueError("katz_centrality: graph is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("katz_centrality: adjacency matrix must be square")
    a = float(alpha)
    b = float(beta)
    if a <= 0:
        raise ValueError("katz_centrality: alpha must be positive")
    K = [[(1.0 if i == j else 0.0) - a * M[i][j] for j in range(n)] for i in range(n)]
    x = [float(v) for v in np.linalg.solve(K, [b] * n)]
    return RichResult(
        title="Katz centrality",
        summary_lines=[("n", n), ("alpha", a), ("beta", b)],
        payload={
            "estimate": max(x),
            "centrality": x,
            "alpha": a,
            "beta": b,
            "n": n,
            "method": "C_K = (I - alpha A)^{-1} beta 1, Katz (1953)",
        },
    )
