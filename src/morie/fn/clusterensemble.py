# morie.fn -- function file (rootcoder007/morie)
"""Resampling-based clustering: Monti consensus clustering, Fred-Jain evidence accumulation and
Hennig's bootstrap cluster stability (Jaccard), all on Lloyd k-means with Philox draws."""

from __future__ import annotations

import bisect

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .clusops import cut_tree, hierarchical_clustering, kmeans_lloyd

__all__ = ["consensus_clustering", "evidence_accumulation", "cluster_stability"]


def _pts(X):
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X]


class _U:
    def __init__(self, seed):
        self.seed, self.block, self.buf, self.pos = seed, 0, [], 0

    def next(self):
        if self.pos >= len(self.buf):
            self.buf = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]

    def index(self, n):
        return min(int(self.next() * n), n - 1)


def _kmeans(P, k, U):
    """Lloyd k-means started at k distinct points drawn uniformly (by index)."""
    n = len(P)
    idx = []
    while len(idx) < k:
        j = U.index(n)
        if all(P[j] != P[i] for i in idx):
            idx.append(j)
    return kmeans_lloyd(P, [P[j] for j in idx]).cluster


def consensus_clustering(X, k: int, *, n_resamples: int = 100, p_item: float = 0.8, seed: int = 1) -> RichResult:
    r"""Consensus clustering (Monti, Tamayo, Mesirov and Golub 2003) with k-means on item subsamples.

    Each of ``n_resamples`` subsamples draws ``round(p_item n)`` items
    without replacement and clusters them; the consensus matrix is
    ``M(i, j) = (times i and j clustered together) / (times both sampled)``.
    The final partition is average-linkage clustering of ``1 - M`` cut at
    ``k``. Also returned: the area under the empirical CDF of the consensus
    values (Monti's ``A(K)``) and the per-cluster consensus ``m(k)`` (mean
    ``M`` within each final cluster).

    References
    ----------
    Monti, S., Tamayo, P., Mesirov, J. and Golub, T. (2003). Consensus
    clustering: a resampling-based method for class discovery and
    visualization of gene expression microarray data. *Machine Learning*,
    52, 91-118.

    Examples
    --------
    >>> X = [[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]]
    >>> consensus_clustering(X, 2, n_resamples=20).cluster
    [1, 1, 1, 2, 2, 2]
    """
    P = _pts(X)
    n = len(P)
    U = _U(seed)
    m = int(round(p_item * n))
    if not k <= m <= n:
        raise ValueError("the subsample must hold at least k items")
    together = [[0] * n for _ in range(n)]
    both = [[0] * n for _ in range(n)]
    for _ in range(n_resamples):
        pool = list(range(n))
        samp = []
        for _t in range(m):
            samp.append(pool.pop(U.index(len(pool))))
        samp.sort()
        lab = _kmeans([P[i] for i in samp], k, U)
        for a in range(m):
            for b in range(m):
                both[samp[a]][samp[b]] += 1
                if lab[a] == lab[b]:
                    together[samp[a]][samp[b]] += 1
    M = [[together[i][j] / both[i][j] if both[i][j] else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    h = hierarchical_clustering([[1.0 - M[i][j] for j in range(n)] for i in range(n)], "average")
    cl = cut_tree(h.merge, k)
    vals = sorted(M[i][j] for i in range(n) for j in range(i + 1, n))
    L = len(vals)
    area = 0.0
    for t in range(1, L):
        area += (vals[t] - vals[t - 1]) * (bisect.bisect_right(vals, vals[t]) / L)
    within = []
    for c in range(1, k + 1):
        mem = [i for i in range(n) if cl[i] == c]
        pairs = [M[a][b] for x, a in enumerate(mem) for b in mem[x + 1 :]]
        within.append(ssum(pairs) / len(pairs) if pairs else 1.0)
    return RichResult(payload={"cluster": cl, "consensus": M, "cdf_area": area, "cluster_consensus": within})


def evidence_accumulation(
    X, k: int | None = None, *, n_runs: int = 50, k_range: tuple = (2, 10), seed: int = 1, linkage: str = "single"
) -> RichResult:
    r"""Evidence accumulation clustering (Fred and Jain 2005): co-association of many k-means partitions.

    ``n_runs`` k-means partitions use a number of clusters drawn uniformly in
    ``k_range`` (capped at ``n``) and random starts; the co-association
    matrix ``C(i, j)`` is the fraction of runs putting ``i`` and ``j``
    together. The final partition cuts the ``linkage`` (default single)
    hierarchy of ``1 - C`` at ``k``, or, when ``k`` is ``None``, at the largest
    lifetime (gap between consecutive merge heights).

    References
    ----------
    Fred, A. L. N. and Jain, A. K. (2005). Combining multiple clusterings
    using evidence accumulation. *IEEE Transactions on Pattern Analysis and
    Machine Intelligence*, 27(6), 835-850.

    Examples
    --------
    >>> X = [[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]]
    >>> evidence_accumulation(X, n_runs=20, k_range=(2, 4)).cluster
    [1, 1, 1, 2, 2, 2]
    """
    P = _pts(X)
    n = len(P)
    U = _U(seed)
    lo, hi = k_range[0], min(k_range[1], n)
    C = [[0.0] * n for _ in range(n)]
    for _ in range(n_runs):
        kk = lo + U.index(hi - lo + 1)
        lab = _kmeans(P, kk, U)
        for i in range(n):
            for j in range(n):
                if lab[i] == lab[j]:
                    C[i][j] += 1.0 / n_runs
    h = hierarchical_clustering([[1.0 - C[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)], linkage)
    if k is None:
        hs = h.height
        gaps = [hs[t + 1] - hs[t] for t in range(len(hs) - 1)]
        t = max(range(len(gaps)), key=lambda q: (gaps[q], -q))
        k = n - (t + 1)
    return RichResult(payload={"cluster": cut_tree(h.merge, k), "coassociation": C, "k": k})


def cluster_stability(X, k: int, *, n_boot: int = 50, seed: int = 1) -> RichResult:
    r"""Bootstrap stability of k-means clusters by the Jaccard coefficient (Hennig 2007, ``fpc::clusterboot``).

    The data are clustered once; in each of ``n_boot`` bootstrap samples
    (drawn with replacement) the sample is re-clustered and every original
    cluster ``C`` restricted to the resampled points is matched with its most
    similar bootstrap cluster by ``|C* & D| / |C* | D|`` (distinct points).
    The mean Jaccard per cluster measures stability (below 0.5: dissolved,
    above 0.75: stable); ``recovered`` counts replicates above 0.5.

    References
    ----------
    Hennig, C. (2007). Cluster-wise assessment of cluster stability.
    *Computational Statistics and Data Analysis*, 52, 258-271.

    Examples
    --------
    >>> X = [[0, 0], [0, 1], [1, 0], [0.5, 0.5], [8, 8], [8, 9], [9, 8], [8.5, 8.5]]
    >>> [round(v, 6) for v in cluster_stability(X, 2, n_boot=10).mean_jaccard]
    [1.0, 1.0]
    """
    P = _pts(X)
    n = len(P)
    U = _U(seed)
    base = _kmeans(P, k, U)
    orig = [{i for i in range(n) if base[i] == c} for c in range(1, k + 1)]
    jac = [[] for _ in range(k)]
    for _ in range(n_boot):
        samp = [U.index(n) for _t in range(n)]
        distinct = sorted(set(samp))
        if len({tuple(P[i]) for i in distinct}) < k:
            continue
        lab = _kmeans([P[i] for i in samp], k, U)
        boot = [{samp[t] for t in range(n) if lab[t] == c} for c in range(1, k + 1)]
        dset = set(distinct)
        for c in range(k):
            cs = orig[c] & dset
            if not cs:
                continue
            best = 0.0
            for D in boot:
                if D:
                    best = max(best, len(cs & D) / len(cs | D))
            jac[c].append(best)
    mean = [ssum(v) / len(v) if v else 0.0 for v in jac]
    return RichResult(
        payload={
            "cluster": base,
            "mean_jaccard": mean,
            "recovered": [sum(1 for x in v if x > 0.5) for v in jac],
            "jaccard": jac,
        }
    )


def cheatsheet() -> str:
    return "consensus_clustering / evidence_accumulation / cluster_stability -> resampling-based clustering."
