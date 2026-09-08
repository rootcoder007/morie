# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Katz centrality excluding the length-zero walk.

Katz (1953), Psychometrika 18(1):39-43, doi:10.1007/BF02289026.  The
variant that drops the k = 0 term of the resolvent series,

    x = (I - alpha A)^{-1} (alpha A) 1 = sum_{k>=1} alpha^k A^k 1,

so a node with no incident walks scores exactly zero instead of one.
The additive constant beta shifts every score by the same amount.
"""

from __future__ import annotations

from . import _array_core as np
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["katz_centrality"]


def katz_centrality(y, A, alpha=0.1, beta=0.0):
    """Walk-weighted status with the self term removed.

    Parameters
    ----------
    y : array-like
        Node weights applied to the all-ones seed vector; pass ones for
        Katz's own definition.
    A, alpha, beta : adjacency matrix, attenuation, additive constant.
    """
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("katz_centrality: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("katz_centrality: adjacency matrix must be square")
    w = core.vec(y)
    if len(w) != n:
        raise ValueError("katz_centrality: y and A have different lengths")
    a = float(alpha)
    if a <= 0:
        raise ValueError("katz_centrality: alpha must be positive")
    rhs = [a * sum(M[i][j] * w[j] for j in range(n)) for i in range(n)]
    K = [[(1.0 if i == j else 0.0) - a * M[i][j] for j in range(n)] for i in range(n)]
    x = [float(v) + float(beta) for v in np.linalg.solve(K, rhs)]
    return RichResult(
        title="Katz centrality (walks of length >= 1)",
        summary_lines=[("n", n), ("alpha", a)],
        payload={
            "estimate": max(x),
            "centrality": x,
            "alpha": a,
            "beta": float(beta),
            "n": n,
            "method": "x = (I - alpha A)^{-1} (alpha A) y + beta, Katz (1953)",
        },
    )
