# morie.fn -- function file (rootcoder007/morie)
"""k-means clustering of compositions in clr coordinates."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compkm", "compositional_kmeans"]


def compkm(X, k=2, max_iter=50):
    """Lloyd k-means on compositions, run in clr coordinates.

    Clustering raw proportions with Euclidean distance is incoherent:
    the answer changes if a part is dropped or the closure constant is
    changed.  Running the same algorithm on clr coordinates makes the
    distance the Aitchison distance, which has neither problem.

    Deterministic by construction, so the two language arms agree
    exactly: the initial centres are the FIRST ``k`` rows, assignment
    ties go to the lowest cluster index, and the iteration stops when
    no label changes or ``max_iter`` is reached.  There is no random
    restart -- callers who want one should supply reordered rows.

    Formula: assign i to argmin_c ||clr(x_i) - m_c||^2; m_c <- mean of
    the clr coordinates assigned to c; centre_c = clr^-1(m_c)

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.
    k : int
        Number of clusters (2 <= k <= n).
    max_iter : int
        Maximum Lloyd sweeps.

    Returns
    -------
    RichResult
        ``cluster`` (one-based labels), ``centers`` (compositions),
        ``clr_centers``, ``withinss``, ``tot_withinss``, ``iterations``,
        ``n``, ``D``, ``k``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 8 (the log-ratio distance being clustered on); Lloyd
    (1982), Least squares quantization in PCM, IEEE Transactions on
    Information Theory 28(2), 129-137 (the algorithm itself).
    """
    X = C.mat(X)
    n = len(X)
    D = len(X[0])
    k = int(k)
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in X]
    Z = [[L[i][j] - sum(L[i]) / D for j in range(D)] for i in range(n)]
    cen = [list(Z[i]) for i in range(k)]
    # -1, not 0: with 0-based labels a first-sweep assignment to cluster 0
    # would not register as a move and the loop would stop a sweep early.
    lab = [-1] * n
    it = 0
    for it in range(1, int(max_iter) + 1):
        moved = False
        for i in range(n):
            best = 0
            bd = None
            for c in range(k):
                d = sum((Z[i][j] - cen[c][j]) ** 2 for j in range(D))
                if bd is None or d < bd:
                    bd = d
                    best = c
            if lab[i] != best:
                lab[i] = best
                moved = True
        for c in range(k):
            mem = [i for i in range(n) if lab[i] == c]
            if mem:
                cen[c] = [sum(Z[i][j] for i in mem) / len(mem) for j in range(D)]
        if not moved:
            break
    wss = [0.0] * k
    for i in range(n):
        wss[lab[i]] += sum((Z[i][j] - cen[lab[i]][j]) ** 2 for j in range(D))
    centers = []
    for c in range(k):
        e = [math.exp(v) for v in cen[c]]
        s = sum(e)
        centers.append([v / s for v in e])
    return RichResult(payload={
        "cluster": [v + 1 for v in lab], "centers": centers,
        "clr_centers": cen, "withinss": wss, "tot_withinss": sum(wss),
        "iterations": it, "n": n, "D": D, "k": k,
        "method": "Lloyd k-means in clr coordinates"})


compositional_kmeans = compkm


def cheatsheet():
    return "aitclu: Lloyd k-means on clr coordinates"
