# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Katz centrality, the resolvent form.

Katz (1953), "A new status index derived from sociometric analysis",
Psychometrika 18(1):39-43, doi:10.1007/BF02289026.  Katz sums walks of
every length with a geometric attenuation a^k, giving

    x = (I - a A)^{-1} 1 = sum_{k>=0} a^k A^k 1,

which converges when a is smaller than the reciprocal of the spectral
radius of A.  The k-th term counts the walks of length k out of each
node, weighted by a^k.
"""

from __future__ import annotations

from . import _array_core as np
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["katz_centrality"]


def _square(A, who):
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError(who + ": adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError(who + ": adjacency matrix must be square")
    return M, n


def katz_centrality(A, alpha=0.1):
    """Solve (I - alpha A) x = 1 for the Katz status scores."""
    M, n = _square(A, "katz_centrality")
    a = float(alpha)
    if a <= 0:
        raise ValueError("katz_centrality: alpha must be positive")
    K = [[(1.0 if i == j else 0.0) - a * M[i][j] for j in range(n)] for i in range(n)]
    x = [float(v) for v in np.linalg.solve(K, [1.0] * n)]
    tot = 0.0
    for v in x:
        tot += v
    return RichResult(
        title="Katz centrality",
        summary_lines=[("n", n), ("alpha", a)],
        payload={
            "estimate": max(x),
            "centrality": x,
            "total": tot,
            "alpha": a,
            "n": n,
            "method": "x = (I - alpha A)^{-1} 1, Katz (1953)",
        },
    )


def cheatsheet():
    return "katz: Katz centrality"


# compact alias per ledger/NAMING.md
katzcentrality = katz_centrality
