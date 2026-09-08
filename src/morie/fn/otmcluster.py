# morie.fn -- function file (rootcoder007/morie)
"""Clustering point clouds under the Wasserstein distance."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_clustering_w"]


def ot_clustering_w(X_list, k, max_iter=10):
    """Lloyd's algorithm with transport in place of Euclidean distance.

    Clustering distributions by comparing their moments throws away shape;
    comparing them bin by bin makes the answer depend on an arbitrary
    binning.  The Wasserstein distance does neither, and the empirical
    version obeys a central limit theorem, which is what makes the
    resulting clusters testable rather than merely descriptive.  Both the
    assignment step and the centroid step are transport problems.

    Formula: alternate ``label(i) = argmin_c W_2(mu_i, nu_c)`` and
    ``nu_c = argmin sum_{i: label(i)=c} W_2^2(mu_i, nu)`` -- the
    barycentre step is Cuturi & Doucet's free-support update.  The
    empirical-transport theory the method rests on is del Barrio et al.
    (2019).

    Parameters
    ----------
    X_list : sequence of arrays, each (n, d)
        Point clouds, all with the same number of points.
    k : int
        Number of clusters.
    max_iter : int, default 10
        Lloyd iterations.

    Returns
    -------
    RichResult
        ``labels``, ``centers``, ``inertia``, ``K``, ``n_clouds``, ``n``,
        ``d``, ``iters``.

    References
    ----------
    del Barrio, E., Cuesta-Albertos, J. A., Matran, C. (2019).  Central
    limit theorems for empirical transportation cost in general
    dimension.  Annals of Probability 47(2):926-951.
    doi:10.1214/18-AOP1275.  Cuturi, M. and Doucet, A. (2014).
    Proceedings of Machine Learning Research 32:685-693 (ICML).
    """
    clouds = [core.mat(X) for X in X_list]
    N = len(clouds)
    if N == 0:
        raise ValueError("no input clouds")
    n = len(clouds[0])
    d = len(clouds[0][0])
    for Xc in clouds:
        if len(Xc) != n or len(Xc[0]) != d:
            raise ValueError("all clouds must have the same shape")
    K = int(k)
    if K < 1 or K > N:
        raise ValueError("k must lie between 1 and the number of clouds")
    u = [1.0 / n] * n
    centers = [[list(r) for r in clouds[c]] for c in range(K)]
    labels = [0] * N
    inertia = 0.0
    it = int(max_iter)
    for _ in range(it):
        inertia = 0.0
        for i in range(N):
            best, bc = None, 0
            for c in range(K):
                _, cost = ot.emd(u, u, ot.costmat(centers[c], clouds[i], 2))
                if best is None or cost < best - 1e-15:
                    best, bc = cost, c
            labels[i] = bc
            inertia += best
        for c in range(K):
            mem = [i for i in range(N) if labels[i] == c]
            if not mem:
                continue
            w = [1.0 / len(mem)] * len(mem)
            Z = [[0.0] * d for _ in range(n)]
            for t, i in enumerate(mem):
                T, _ = ot.emd(u, u, ot.costmat(centers[c], clouds[i], 2))
                for r in range(n):
                    for s in range(n):
                        if T[r][s] == 0.0:
                            continue
                        for q in range(d):
                            Z[r][q] += w[t] * n * T[r][s] * clouds[i][s][q]
            centers[c] = Z
    return RichResult(payload={
        "labels": labels, "centers": centers, "inertia": inertia,
        "K": K, "n_clouds": N, "n": n, "d": d, "iters": it,
        "method": "Wasserstein k-means over point clouds"})


def cheatsheet():
    return "otmcluster: Wasserstein k-means clustering of point clouds"


# compact alias per ledger/NAMING.md
otclusteringw = ot_clustering_w
