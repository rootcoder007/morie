# morie.fn -- slice s04 (rootcoder007/morie)
"""Principal component based GxE dimension reduction.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 35-70], Chapter 2, Section 2.8
"Principal Component Analysis as a Compression Method", pp. 63-68.  The
section gives PC = XW, "where W is a p-by-p matrix of weights whose
columns are the eigenvectors of Q = X'X, that is, we first need to
calculate the eigenvalue decomposition of Q, which is equal to
Q = W Lambda W'"; the compressed matrix is X* = XW*, "where W* contains
the same rows of W, but only the first k columns".  It adds that "the
principal components can be obtained from a covariance matrix,
Q = 1/(n-1) X'X, where each column of X is centered, or from the
correlation matrix ... where each column of the original matrix of
information was standardized", and its worked example prints the
component standard deviations and the leading scores for the fifteen-line
data of Table 2.13.  Those printed numbers are the anchor.

The docstring's SVD form GxE = U_k D_k V_k' is the same decomposition:
the eigenvectors of X'X are V and the scores XV are U D.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["pc_gxe_reduction"]


def pc_gxe_reduction(GxE_matrix, k, scale=False):
    """Rank-k principal component reduction of a GxE matrix.

    Parameters
    ----------
    GxE_matrix : array-like
        n-by-p matrix; columns are centred (and optionally scaled) first.
    k : int
        Number of components to retain, 1 <= k <= p.
    scale : bool
        Divide each centred column by its sample standard deviation, so
        Q is the correlation matrix rather than the covariance matrix.

    Returns
    -------
    estimate     : the proportion of variance the k components explain
    scores       : the n-by-k matrix XW*
    loadings     : the p-by-k matrix W*
    sdev         : the component standard deviations, descending
    eigenvalues  : the eigenvalues of Q, descending
    GxE_approx   : the rank-k reconstruction XW*W*'
    """
    X = core.mat(GxE_matrix)
    n = len(X)
    if n < 2:
        raise ValueError("pc_gxe_reduction: need at least two rows")
    p = len(X[0])
    for r in X:
        if len(r) != p:
            raise ValueError("pc_gxe_reduction: rows have unequal lengths")
    kk = int(k)
    if kk < 1 or kk > p:
        raise ValueError("pc_gxe_reduction: k must lie between 1 and the number of columns")
    mu = []
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i][j]
        mu.append(s / n)
    C = [[X[i][j] - mu[j] for j in range(p)] for i in range(n)]
    sc = [1.0] * p
    if scale:
        for j in range(p):
            s = 0.0
            for i in range(n):
                s += C[i][j] * C[i][j]
            s = math.sqrt(s / (n - 1))
            if s <= 0.0:
                raise ValueError("pc_gxe_reduction: a column has zero variance and cannot be scaled")
            sc[j] = s
            for i in range(n):
                C[i][j] = C[i][j] / s
    Q = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(p):
            s = 0.0
            for i in range(n):
                s += C[i][a] * C[i][b]
            Q[a][b] = s / (n - 1)
    val, vecs = core.jacobi(Q)
    order = list(range(p - 1, -1, -1))          # jacobi returns ascending
    ev = [val[o] for o in order]
    W = [[vecs[r][order[j]] for j in range(kk)] for r in range(p)]
    scores = [[sum(C[i][a] * W[a][j] for a in range(p)) for j in range(kk)] for i in range(n)]
    approx = [[sum(scores[i][j] * W[a][j] for j in range(kk)) for a in range(p)] for i in range(n)]
    tot = 0.0
    for v in ev:
        tot += v
    top = 0.0
    for j in range(kk):
        top += ev[j]
    return RichResult(
        title="Principal component GxE reduction",
        summary_lines=[("rows", n), ("columns", p), ("kept", kk)],
        payload={
            "estimate": top / tot if tot > 0.0 else float("nan"),
            "scores": scores,
            "loadings": W,
            "sdev": [math.sqrt(v) if v > 0.0 else 0.0 for v in ev],
            "eigenvalues": ev,
            "GxE_approx": approx,
            "center": mu,
            "scale": sc,
            "n": n,
            "method": "GxE = U_k D_k V_k' by the eigendecomposition of Q, Chapter 2 Sect. 2.8",
        },
    )


def cheatsheet():
    return "pcgxe: Principal component based GxE dimension reduction"


# compact alias per ledger/NAMING.md
pcgxereduction = pc_gxe_reduction
