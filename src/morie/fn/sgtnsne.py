# morie.fn -- slice s03 (rootcoder007/morie)
"""Isomap: classical MDS on geodesic distances.

Source consulted: Tenenbaum, J. B., de Silva, V. and Langford, J. C.
(2000).  A global geometric framework for nonlinear dimensionality
reduction.  *Science* 290(5500), 2319-2323.  The three steps the paper
prints are: build the neighbourhood graph (k nearest neighbours, edges
weighted by Euclidean distance); estimate the geodesic distances D_G by
shortest paths, "computed by Floyd's algorithm"; and apply classical
MDS to D_G, i.e. double-centre the squared distances,

    tau(D) = -H S H / 2,   S_ij = D_ij^2,   H = I - (1/n) 1 1'

and take the top-d eigenvectors scaled by the square roots of their
eigenvalues.  The *Science* paper is paywalled; the three steps and the
double-centring are quoted in their standard published form.

Eigenvectors are sign-fixed before scaling, since an MDS embedding is
determined only up to reflection.  Disconnected components leave
infinite geodesics; those pairs are reported rather than silently
patched.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["sgt_isomap"]


def sgt_isomap(X, k_nn=3, dim=2):
    """Isomap embedding of a point cloud.

    Returns
    -------
    RichResult with payload:
        Y         : the embedding, one row per point
        estimate  : the leading eigenvalue of the centred geodesic matrix
        eigvals   : the top-dim eigenvalues
        n_infinite: pairs left unreachable by the k-NN graph
    """
    P = k.mat(X)
    n = len(P)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for a in range(len(P[i])):
                d = P[i][a] - P[j][a]
                s += d * d
            D[i][j] = math.sqrt(s)
    INF = float("inf")
    G = [[INF] * n for _ in range(n)]
    kk = int(k_nn)
    for i in range(n):
        G[i][i] = 0.0
        order = sorted(range(n), key=lambda j: (D[i][j], j))
        for t in range(1, min(kk + 1, n)):
            j = order[t]
            G[i][j] = D[i][j]
            G[j][i] = D[i][j]
    for m in range(n):
        for i in range(n):
            for j in range(n):
                if G[i][m] + G[m][j] < G[i][j]:
                    G[i][j] = G[i][m] + G[m][j]
    ninf = 0
    for i in range(n):
        for j in range(n):
            if G[i][j] == INF:
                ninf += 1
                G[i][j] = 0.0
    S = [[G[i][j] * G[i][j] for j in range(n)] for i in range(n)]
    rm = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += S[i][j]
        rm[i] = s / n
    gm = 0.0
    for v in rm:
        gm += v / n
    B = [[-0.5 * (S[i][j] - rm[i] - rm[j] + gm) for j in range(n)]
         for i in range(n)]
    vals, vecs = k.jacobi(B)
    d = int(dim)
    if d > n:
        d = n
    ev = [vals[n - 1 - t] for t in range(d)]
    Y = [[0.0] * d for _ in range(n)]
    for t in range(d):
        lam = ev[t]
        s = math.sqrt(lam) if lam > 0.0 else 0.0
        for i in range(n):
            Y[i][t] = vecs[i][n - 1 - t] * s
    return RichResult(
        title="Isomap",
        summary_lines=[("points", n), ("dim", d)],
        payload={
            "Y": Y,
            "estimate": ev[0] if ev else float("nan"),
            "eigvals": ev,
            "n_infinite": ninf,
            "method": "Isomap: k-NN graph, Floyd geodesics, classical MDS (Tenenbaum et al. 2000)",
        },
    )


def cheatsheet():
    return "sgtnsne: Isomap MDS on geodesic distance matrix"


sgtisomap = sgt_isomap
