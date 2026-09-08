# morie.fn -- k02 batch (rootcoder007/morie)
"""Small-world coefficient S.

Source consulted: Humphries, M.D. and Gurney, K. (2008), Network
"small-world-ness": a quantitative method for determining canonical network
equivalence, *PLoS ONE* 3(4), e0002051, equations (1)-(4).  With C the mean
local clustering coefficient and L the characteristic path length,

    S = (C / C_rand) / (L / L_rand),
    C_rand = kbar / n,   L_rand = ln(n) / ln(kbar)

using the paper's analytic Erdos-Renyi reference rather than a simulated one,
so the statistic is deterministic.  S > 1 is the paper's criterion for
small-worldness.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02bfs

from ._richresult import RichResult

__all__ = ["small_worldness"]


def small_worldness(A):
    """Humphries-Gurney small-world coefficient.

    Parameters
    ----------
    A : array-like
        Symmetric binary adjacency matrix.

    Returns
    -------
    RichResult
        estimate (S), clustering, path_length, clustering_random,
        path_length_random, mean_degree, n, method.
    """
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    b = (a != 0.0)
    deg = [int(sum(1 for j in range(n) if j != i and a[i, j] != 0.0)) for i in range(n)]
    cl = []
    for i in range(n):
        nb = [j for j in range(n) if j != i and a[i, j] != 0.0]
        k = len(nb)
        if k < 2:
            cl.append(0.0)
            continue
        links = 0
        for p in range(k):
            for q in range(p + 1, k):
                if a[nb[p], nb[q]] != 0.0:
                    links += 1
        cl.append(2.0 * links / (k * (k - 1)))
    cbar = float(np.mean(np.asarray(cl, dtype=float)))
    d = k02bfs(a)
    tot = 0.0
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i != j and d[i][j] >= 0:
                tot += d[i][j]
                cnt += 1
    lbar = tot / cnt if cnt else float("nan")
    kbar = float(np.mean(np.asarray(deg, dtype=float)))
    crand = kbar / n
    lrand = float(np.log(n)) / float(np.log(kbar)) if kbar > 1.0 else float("nan")
    s = (cbar / crand) / (lbar / lrand)
    return RichResult(
        payload={
            "estimate": float(s),
            "clustering": cbar,
            "path_length": float(lbar),
            "clustering_random": float(crand),
            "path_length_random": float(lrand),
            "mean_degree": kbar,
            "n": int(n),
            "method": "Small-world coefficient S (Humphries & Gurney 2008, eq. 1-4)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> r = small_worldness(A)
# >>> assert abs(r["clustering"] - 0.777777777777778) < 1e-12   # igraph transitivity
# >>> assert abs(r["path_length"] - 1.8) < 1e-12                # igraph mean_distance
# >>> assert r["estimate"] > 1.0


def cheatsheet():
    return "smwgrp(A): Humphries-Gurney small-world coefficient."


smallworldness = small_worldness
