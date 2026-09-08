# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""GCNII: initial residual plus identity mapping.

Chen, Wei, Huang, Ding and Li (2020), "Simple and deep graph
convolutional networks", ICML 2020 (PMLR 119:1725-1735),
arXiv:2007.02133, equation (3):

    H^{l+1} = sigma( ((1-a) Ah H^l + a H^0) ((1-b_l) I + b_l W^l) ).

The first bracket is the initial residual connection, the second the
identity mapping.  With W^l taken as the identity -- the case the
paper's own analysis calls the "identity mapping" limit, and the only
deterministic choice available here since no weight is passed -- the
second bracket collapses to I for every b, so the recursion reduces to
sigma((1-a) Ah H + a H^0).  Ah is the renormalised operator
Dt^{-1/2}(A+I)Dt^{-1/2}.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gcnii"]


def gcnii(A, H0, alpha=0.1, beta=0.5, K=4):
    """K layers of the GCNII recursion with W = I."""
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("gcnii: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("gcnii: adjacency matrix must be square")
    H = core.mat(H0)
    if len(H) != n:
        raise ValueError("gcnii: H0 must have one row per node")
    a = float(alpha)
    if a < 0 or a > 1:
        raise ValueError("gcnii: alpha must lie in [0, 1]")
    if float(beta) < 0:
        raise ValueError("gcnii: beta must be non-negative")
    layers = int(K)
    if layers < 1:
        raise ValueError("gcnii: K must be at least 1")
    At = [[M[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    d = [sum(At[i]) for i in range(n)]
    s = [0.0 if d[i] <= 0 else d[i] ** -0.5 for i in range(n)]
    Ah = [[s[i] * At[i][j] * s[j] for j in range(n)] for i in range(n)]
    Hl = [row[:] for row in H]
    for _ in range(layers):
        P = core.matmul(Ah, Hl)
        Hl = [[core.relu((1.0 - a) * P[i][j] + a * H[i][j]) for j in range(len(H[0]))] for i in range(n)]
    flat = [v for row in Hl for v in row]
    return RichResult(
        title="GCNII",
        summary_lines=[("nodes", n), ("layers", layers), ("alpha", a)],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": Hl,
            "alpha": a,
            "beta": float(beta),
            "K": layers,
            "n": n,
            "method": "H = relu((1-a) Ah H + a H0) with W = I, Chen et al. (2020) eq. (3)",
        },
    )
