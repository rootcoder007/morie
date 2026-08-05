# morie.fn -- function file (rootcoder007/morie)
"""Distance-dependent Chinese restaurant process."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["clustered_crp"]


def clustered_crp(y, distances, alpha=1.0, decay=1.0, seed=42):
    """
    Distance-dependent CRP

    Formula: P(c_i = j) proportional to f(d_ij) for j != i, alpha for j = i

    Customers link to other CUSTOMERS rather than to tables, and the
    clusters are the connected components of the link graph.  With the
    exponential decay f(d) = exp(-d/decay), a decay near zero makes
    every customer link to itself and gives n singleton clusters, while
    a very large decay makes every link equally likely and collapses the
    partition.  Unlike the ordinary CRP the induced partition is not
    exchangeable, which is the point: distance carries information.

    Parameters
    ----------
    y : array-like
        Observations, used only for the reported cluster means.
    distances : array-like
        n x n matrix of pairwise distances.
    alpha : float
        Self-link weight, strictly positive.
    decay : float
        Scale of the exponential decay, strictly positive.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (number of clusters), z, links, counts,
        cluster_mean, n_clusters, n.

    References
    ----------
    Blei & Frazier (2011), J. Machine Learning Research 12:2461-2488.
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    D = core.mat(distances)
    if len(D) != n or any(len(r) != n for r in D):
        raise ValueError("distances must be an n x n matrix")
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    if not (decay > 0.0):
        raise ValueError("decay must be strictly positive")
    rng = np.random.default_rng(seed)
    links = [0] * n
    for i in range(n):
        w = []
        for j in range(n):
            w.append(alpha if j == i else math.exp(-D[i][j] / decay))
        tot = sum(w)
        u = float(rng.uniform(0.0, 1.0)) * tot
        acc = 0.0
        pick = n - 1
        for j in range(n):
            acc += w[j]
            if u <= acc:
                pick = j
                break
        links[i] = pick
    # connected components of the undirected link graph
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        ra, rb = find(i), find(links[i])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = []
    z = [0] * n
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots.append(r)
        z[i] = roots.index(r)
    K = len(roots)
    counts = [sum(1 for v in z if v == c) for c in range(K)]
    means = [sum(y[i] for i in range(n) if z[i] == c) / counts[c]
             for c in range(K)]
    return RichResult(payload={
        "estimate": K,
        "z": z,
        "links": links,
        "counts": counts,
        "cluster_mean": means,
        "n_clusters": K,
        "n": n,
        "method": "distance-dependent Chinese restaurant process",
    })


def cheatsheet():
    return "clcrp: distance-dependent Chinese restaurant process"


# compact alias per ledger/NAMING.md
clusteredcrp = clustered_crp
