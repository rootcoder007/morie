# morie.fn -- function file (rootcoder007/morie)
"""Partitioning, hierarchical, density-based and fuzzy clustering with validity indices."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "kmeans_lloyd",
    "pam_medoids",
    "silhouette_widths",
    "dunn_index",
    "hierarchical_clustering",
    "cut_tree",
    "dbscan_clusters",
    "optics_ordering",
    "fuzzy_cmeans",
    "mean_shift",
]


def _pts(X):
    P = [[float(v) for v in (r if isinstance(r, list) else [r])] for r in np.asarray(X, dtype=float).tolist()]
    if not P:
        raise ValueError("X must have rows")
    return P


def _d(a, b):
    return math.sqrt(ssum((x - y) ** 2 for x, y in zip(a, b)))


def _dm(X):
    P = _pts(X)
    return [[_d(a, b) for b in P] for a in P]


def kmeans_lloyd(X, centers, *, maxit: int = 100) -> RichResult:
    r"""Lloyd's k-means from given initial centres, as ``kmeans(algorithm = "Lloyd")``.

    Alternates assigning each point to its nearest centre (first on ties)
    and moving each centre to the mean of its points, until the assignment
    no longer changes (Lloyd 1982).  Returns ``cluster`` (1-based),
    ``centers``, ``withinss``, ``tot_withinss``, ``betweenss``, ``iter``.

    References
    ----------
    Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE
    Transactions on Information Theory*, 28(2), 129-137.

    Examples
    --------
    >>> r = kmeans_lloyd([[0, 0], [0, 1], [5, 5], [5, 6]], [[0, 0], [5, 5]])
    >>> r.cluster, r.tot_withinss
    ([1, 1, 2, 2], 1.0)
    """
    P = _pts(X)
    C = _pts(centers)
    k, p = len(C), len(P[0])
    cl = None
    it = 0
    for _ in range(maxit):
        it += 1
        new = [min(range(k), key=lambda j: (ssum((a - b) ** 2 for a, b in zip(x, C[j])), j)) for x in P]
        if new == cl:
            break
        cl = new
        for j in range(k):
            m = [x for x, c in zip(P, cl) if c == j]
            if m:
                C[j] = [ssum(x[t] for x in m) / len(m) for t in range(p)]
    wss = [ssum(ssum((a - b) ** 2 for a, b in zip(x, C[j])) for x, c in zip(P, cl) if c == j) for j in range(k)]
    mu = [ssum(x[t] for x in P) / len(P) for t in range(p)]
    tss = ssum(ssum((a - b) ** 2 for a, b in zip(x, mu)) for x in P)
    return RichResult(
        payload={
            "cluster": [c + 1 for c in cl],
            "centers": C,
            "withinss": wss,
            "tot_withinss": ssum(wss),
            "betweenss": tss - ssum(wss),
            "iter": it,
        }
    )


def pam_medoids(D, k: int, *, is_distance: bool = True) -> RichResult:
    r"""Partitioning around medoids (Kaufman and Rousseeuw 1990), BUILD then SWAP, as ``cluster::pam``.

    BUILD selects the object with the smallest total dissimilarity, then
    repeatedly the object that most decreases the objective; SWAP performs
    the medoid/non-medoid exchange with the largest decrease of the total
    dissimilarity to the nearest medoid until none decreases it.  Returns
    ``medoids`` (1-based), ``clustering``, ``objective`` (average
    dissimilarity after BUILD and after SWAP); clusters are numbered by
    first appearance in the observations and ``medoids`` follow that order.

    References
    ----------
    Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data: An
    Introduction to Cluster Analysis*. Wiley, New York, chapter 2.

    Examples
    --------
    >>> r = pam_medoids([[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]], 2)
    >>> r.medoids, r.clustering
    ([2, 3], [1, 1, 2, 2])
    """
    M = [[float(v) for v in r] for r in D] if is_distance else _dm(D)
    n = len(M)
    if not 1 <= k < n:
        raise ValueError("k must be in 1..n-1")

    def cost(meds):
        return ssum(min(M[i][m] for m in meds) for i in range(n))

    meds = [min(range(n), key=lambda i: (ssum(M[i]), i))]
    while len(meds) < k:
        cur = cost(meds)
        best = max((i for i in range(n) if i not in meds), key=lambda i: (cur - cost(meds + [i]), -i))
        meds.append(best)
    build = cost(meds) / n
    while True:
        cur = cost(meds)
        best, bgain = None, 1e-10 * max(1.0, cur)
        for a in range(k):
            for h in range(n):
                if h in meds:
                    continue
                trial = meds[:a] + [h] + meds[a + 1 :]
                g = cur - cost(trial)
                if g > bgain:
                    best, bgain = trial, g
        if best is None:
            break
        meds = best
    near = [min(range(k), key=lambda j: (M[i][meds[j]], j)) for i in range(n)]
    # clusters numbered by first appearance in the observations, medoids in that order (cluster::pam)
    first = []
    for j in near:
        if j not in first:
            first.append(j)
    meds = [meds[j] for j in first]
    clus = [first.index(j) + 1 for j in near]
    return RichResult(
        payload={
            "medoids": [m + 1 for m in meds],
            "clustering": clus,
            "objective": {"build": build, "swap": cost(meds) / n},
        }
    )


def silhouette_widths(D, clusters, *, is_distance: bool = True) -> RichResult:
    r"""Silhouette widths ``s(i) = (b_i - a_i) / max(a_i, b_i)`` (Rousseeuw 1987), as ``cluster::silhouette``.

    ``a_i`` is the mean dissimilarity to the other members of its cluster,
    ``b_i`` the smallest mean dissimilarity to another cluster (whose label
    is returned as ``neighbor``); singletons get ``s = 0``.

    References
    ----------
    Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
    interpretation and validation of cluster analysis. *Journal of
    Computational and Applied Mathematics*, 20, 53-65.

    Examples
    --------
    >>> s = silhouette_widths([[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]], [1, 1, 2, 2])
    >>> [round(v, 6) for v in s.width]
    [0.818182, 0.777778, 0.777778, 0.818182]
    """
    M = [[float(v) for v in r] for r in D] if is_distance else _dm(D)
    lab = list(clusters)
    n = len(M)
    cls = sorted(set(lab))
    width, neigh = [], []
    for i in range(n):
        own = [j for j in range(n) if lab[j] == lab[i] and j != i]
        if not own:
            width.append(0.0)
            others = {
                c: ssum(M[i][j] for j in range(n) if lab[j] == c) / sum(1 for j in range(n) if lab[j] == c)
                for c in cls
                if c != lab[i]
            }
            neigh.append(min(others, key=lambda c: (others[c], cls.index(c))))
            continue
        a = ssum(M[i][j] for j in own) / len(own)
        others = {
            c: ssum(M[i][j] for j in range(n) if lab[j] == c) / sum(1 for j in range(n) if lab[j] == c)
            for c in cls
            if c != lab[i]
        }
        nb = min(others, key=lambda c: (others[c], cls.index(c)))
        b = others[nb]
        width.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
        neigh.append(nb)
    avg = {c: ssum(w for w, g in zip(width, lab) if g == c) / sum(1 for g in lab if g == c) for c in cls}
    return RichResult(payload={"width": width, "neighbor": neigh, "cluster_avg": avg, "avg_width": ssum(width) / n})


def dunn_index(D, clusters, *, is_distance: bool = True) -> float:
    r"""Dunn index: smallest between-cluster distance over the largest cluster diameter (Dunn 1974).

    References
    ----------
    Dunn, J. C. (1974). Well-separated clusters and optimal fuzzy
    partitions. *Journal of Cybernetics*, 4(1), 95-104.

    Examples
    --------
    >>> dunn_index([[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]], [1, 1, 2, 2])
    4.0
    """
    M = [[float(v) for v in r] for r in D] if is_distance else _dm(D)
    lab = list(clusters)
    n = len(M)
    inter = min(M[i][j] for i in range(n) for j in range(n) if lab[i] != lab[j])
    diam = max((M[i][j] for i in range(n) for j in range(n) if lab[i] == lab[j]), default=0.0)
    return inter / diam if diam > 0 else float("inf")


def hierarchical_clustering(D, method: str = "complete", *, is_distance: bool = True) -> RichResult:
    r"""Agglomerative hierarchical clustering, merge matrix and heights as ``stats::hclust``.

    At each step the two closest clusters merge; dissimilarities are updated
    by the Lance-Williams formula: ``single`` (min), ``complete`` (max),
    ``average`` (UPGMA, size-weighted mean) or ``ward.D2`` (Ward's criterion
    on squared distances, heights on the distance scale; Murtagh and
    Legendre 2014).  ``merge`` rows follow the R convention: negative
    entries are singletons, positive ones earlier merges; a row lists a
    singleton before a cluster, two singletons or two clusters in increasing
    order.

    References
    ----------
    Lance, G. N. and Williams, W. T. (1967). A general theory of
    classificatory sorting strategies: 1. Hierarchical systems. *The
    Computer Journal*, 9(4), 373-380.
    Murtagh, F. and Legendre, P. (2014). Ward's hierarchical agglomerative
    clustering method: which algorithms implement Ward's criterion?
    *Journal of Classification*, 31(3), 274-295.

    Examples
    --------
    >>> h = hierarchical_clustering([[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]], "single")
    >>> h.merge, h.height
    ([[-1, -2], [-3, -4], [1, 2]], [1.0, 1.0, 4.0])
    """
    if method not in ("single", "complete", "average", "ward.D2"):
        raise ValueError("method must be single, complete, average or ward.D2")
    M = [[float(v) for v in r] for r in D] if is_distance else _dm(D)
    n = len(M)
    ward = method == "ward.D2"
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = M[i][j] ** 2 if ward else M[i][j]
    ids = {i: -(i + 1) for i in range(n)}  # active cluster -> R label
    size = {i: 1 for i in range(n)}
    active = list(range(n))
    merge, height = [], []
    for step in range(1, n):
        (a, b) = min(
            ((x, y) for ix, x in enumerate(active) for y in active[ix + 1 :]),
            key=lambda t: (dist[(min(t), max(t))], min(t), max(t)),
        )
        a, b = min(a, b), max(a, b)
        dab = dist[(a, b)]
        la, lb = ids[a], ids[b]
        if la < 0 and lb < 0:
            row = sorted([la, lb], reverse=True)
        elif la < 0 or lb < 0:
            row = [min(la, lb), max(la, lb)]
        else:
            row = sorted([la, lb])
        merge.append(row)
        height.append(math.sqrt(dab) if ward else dab)
        na, nb = size[a], size[b]
        for c in active:
            if c in (a, b):
                continue
            dac, dbc = dist[(min(a, c), max(a, c))], dist[(min(b, c), max(b, c))]
            if method == "single":
                v = min(dac, dbc)
            elif method == "complete":
                v = max(dac, dbc)
            elif method == "average":
                v = (na * dac + nb * dbc) / (na + nb)
            else:
                nc = size[c]
                v = ((na + nc) * dac + (nb + nc) * dbc - nc * dab) / (na + nb + nc)
            dist[(min(a, c), max(a, c))] = v
        active.remove(b)
        size[a] = na + nb
        ids[a] = step
    return RichResult(payload={"merge": merge, "height": height, "method": method, "n": n})


def cut_tree(merge, k: int):
    r"""Cluster labels for ``k`` groups from an R-style merge matrix, numbered by first appearance (``cutree``).

    Examples
    --------
    >>> cut_tree([[-1, -2], [-3, -4], [1, 2]], 2)
    [1, 1, 2, 2]
    """
    n = len(merge) + 1
    if not 1 <= k <= n:
        raise ValueError("k must be in 1..n")
    g = [-(i + 1) for i in range(n)]  # current top-level label: -i singleton, s merge step
    for s, (a, b) in enumerate(merge[: n - k], start=1):
        g = [s if v in (a, b) else v for v in g]
    seen = []
    for v in g:
        if v not in seen:
            seen.append(v)
    return [seen.index(v) + 1 for v in g]


def dbscan_clusters(X, eps: float, min_pts: int = 5, *, border_points: bool = True) -> RichResult:
    r"""DBSCAN (Ester et al. 1996), labels as ``dbscan::dbscan`` (0 = noise).

    Core points have at least ``min_pts`` points (themselves included)
    within distance ``eps``; clusters are grown from core points in index
    order, each reachable border point joining the first cluster that
    reaches it (dropped to noise without ``border_points``).

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J. and Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. *KDD-96*, 226-231.

    Examples
    --------
    >>> dbscan_clusters([[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9], [5, 5]], 1.5, 3).cluster
    [1, 1, 1, 2, 2, 2, 0]
    """
    P = _pts(X)
    n = len(P)
    nb = [[j for j in range(n) if _d(P[i], P[j]) <= eps] for i in range(n)]
    core = [len(v) >= min_pts for v in nb]
    lab = [0] * n
    c = 0
    for i in range(n):
        if lab[i] or not core[i]:
            continue
        c += 1
        lab[i] = c
        queue = [i]
        while queue:
            q = queue.pop(0)
            if not core[q]:
                continue
            for j in nb[q]:
                if lab[j] == 0:
                    if core[j] or border_points:
                        lab[j] = c
                    if core[j]:
                        queue.append(j)
    return RichResult(payload={"cluster": lab, "core": core})


def optics_ordering(X, eps: float = float("inf"), min_pts: int = 5) -> RichResult:
    r"""OPTICS cluster ordering (Ankerst et al. 1999), as ``dbscan::optics``.

    ``coredist`` is the distance to the ``min_pts``-th nearest point (itself
    included; ``inf`` beyond ``eps``); points are expanded in index order of
    unprocessed seeds, always taking the seed with the smallest reachability
    ``max(coredist(p), d(p, o))`` (ties by index).  Returns ``order``
    (1-based), ``reachdist`` and ``coredist`` indexed by point (``inf`` for
    undefined).

    References
    ----------
    Ankerst, M., Breunig, M. M., Kriegel, H.-P. and Sander, J. (1999).
    OPTICS: ordering points to identify the clustering structure. *ACM
    SIGMOD Record*, 28(2), 49-60.

    Examples
    --------
    >>> o = optics_ordering([[0, 0], [0, 1], [0, 3], [10, 0]], 5, 2)
    >>> o.order, o.coredist
    ([1, 2, 3, 4], [1.0, 1.0, 2.0, inf])
    """
    P = _pts(X)
    n = len(P)
    inf = float("inf")
    D = [[_d(a, b) for b in P] for a in P]
    core = []
    for i in range(n):
        ds = sorted(D[i])
        cd = ds[min_pts - 1] if len(ds) >= min_pts else inf
        core.append(cd if cd <= eps else inf)
    reach = [inf] * n
    done = [False] * n
    order = []
    for s in range(n):
        if done[s]:
            continue
        seeds = {}
        cur = s
        while True:
            done[cur] = True
            order.append(cur + 1)
            if core[cur] < inf:
                for j in range(n):
                    if not done[j] and D[cur][j] <= eps:
                        r = max(core[cur], D[cur][j])
                        if r < reach[j]:
                            reach[j] = r
                            seeds[j] = r
            if not seeds:
                break
            cur = min(seeds, key=lambda j: (seeds[j], -j))
            del seeds[cur]
    return RichResult(payload={"order": order, "reachdist": reach, "coredist": core})


def fuzzy_cmeans(X, centers, *, m: float = 2.0, tol: float = 1e-9, maxit: int = 1000) -> RichResult:
    r"""Fuzzy c-means (Bezdek 1981) from given initial centres, with the Xie-Beni index.

    Memberships ``u_ik = 1 / sum_j (d_ik / d_ij)^{2/(m-1)}`` and centres
    ``v_k = sum_i u_ik^m x_i / sum_i u_ik^m`` alternate until the centres
    move less than ``tol``; ``xie_beni = sum u^m ||x - v||^2 / (n min_{k != l}
    ||v_k - v_l||^2)`` (Xie and Beni 1991); ``partition_coefficient`` ``sum
    u^2 / n``.

    References
    ----------
    Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function
    Algorithms*. Plenum, New York.
    Xie, X. L. and Beni, G. (1991). A validity measure for fuzzy clustering.
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(8),
    841-847.

    Examples
    --------
    >>> r = fuzzy_cmeans([[0.0], [1.0], [9.0], [10.0]], [[0.0], [10.0]])
    >>> [round(v[0], 6) for v in r.centers]
    [0.49974, 9.50026]
    """
    P = _pts(X)
    V = _pts(centers)
    n, k, p = len(P), len(V), len(P[0])
    if m <= 1:
        raise ValueError("m must exceed 1")
    e = 2.0 / (m - 1.0)
    U = []
    it = 0
    for _ in range(maxit):
        it += 1
        U = []
        for x in P:
            d = [_d(x, v) for v in V]
            if any(v == 0 for v in d):
                U.append([1.0 if v == 0 else 0.0 for v in d])
                s = sum(U[-1])
                U[-1] = [u / s for u in U[-1]]
            else:
                U.append([1.0 / ssum((d[a] / d[b]) ** e for b in range(k)) for a in range(k)])
        newV = [
            [ssum(U[i][a] ** m * P[i][t] for i in range(n)) / ssum(U[i][a] ** m for i in range(n)) for t in range(p)]
            for a in range(k)
        ]
        shift = max(_d(a, b) for a, b in zip(V, newV))
        V = newV
        if shift < tol:
            break
    J = ssum(U[i][a] ** m * _d(P[i], V[a]) ** 2 for i in range(n) for a in range(k))
    sep = min(_d(V[a], V[b]) ** 2 for a in range(k) for b in range(a + 1, k))
    return RichResult(
        payload={
            "centers": V,
            "membership": U,
            "objective": J,
            "xie_beni": J / (n * sep),
            "partition_coefficient": ssum(u * u for r in U for u in r) / n,
            "cluster": [max(range(k), key=lambda a: (r[a], -a)) + 1 for r in U],
            "iter": it,
        }
    )


def mean_shift(
    X, bandwidth: float, *, tol: float = 1e-10, maxit: int = 1000, merge_tol: float | None = None
) -> RichResult:
    r"""Gaussian mean-shift mode seeking (Fukunaga and Hostetler 1975; Comaniciu and Meer 2002).

    Every point climbs ``y <- sum_i K(y - x_i) x_i / sum_i K(y - x_i)`` with
    ``K(u) = exp(-||u||^2 / (2 h^2))`` until it moves less than ``tol``;
    modes closer than ``merge_tol`` (default ``h / 100``) are merged in
    order of first appearance.

    References
    ----------
    Comaniciu, D. and Meer, P. (2002). Mean shift: a robust approach toward
    feature space analysis. *IEEE Transactions on Pattern Analysis and
    Machine Intelligence*, 24(5), 603-619.

    Examples
    --------
    >>> mean_shift([[0.0], [0.2], [5.0], [5.2]], 0.5).cluster
    [1, 1, 2, 2]
    """
    P = _pts(X)
    n, p = len(P), len(P[0])
    mt = bandwidth / 100.0 if merge_tol is None else merge_tol
    ends = []
    for x in P:
        y = list(x)
        for _ in range(maxit):
            w = [math.exp(-ssum((a - b) ** 2 for a, b in zip(y, q)) / (2.0 * bandwidth**2)) for q in P]
            sw = ssum(w)
            ny = [ssum(w[i] * P[i][t] for i in range(n)) / sw for t in range(p)]
            mv = _d(ny, y)
            y = ny
            if mv < tol:
                break
        ends.append(y)
    modes, lab = [], []
    for y in ends:
        for c, mo in enumerate(modes):
            if _d(y, mo) < mt:
                lab.append(c + 1)
                break
        else:
            modes.append(y)
            lab.append(len(modes))
    return RichResult(payload={"cluster": lab, "modes": modes, "endpoints": ends})


def cheatsheet() -> str:
    return "kmeans_lloyd / pam_medoids / hierarchical_clustering / dbscan_clusters / optics_ordering -> clustering."
