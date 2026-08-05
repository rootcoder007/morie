# morie.fn -- function file (rootcoder007/morie)
"""Within-cluster standardisation (cluster z-score)."""

import math

from . import _tail1core as C
from .cwcm import centering_within_cluster_mean

from ._richresult import RichResult

__all__ = ["multilevel_within_cluster_z", "multilevelwithinclusterz"]


def multilevel_within_cluster_z(y, cluster, ddof=1):
    """Standardise a level-1 variable inside each cluster.

        z_ij = (x_ij - xbar_j) / sd_j

    The centring step is not recomputed here: it is
    ``cwcm.centering_within_cluster_mean``, the single implementation of
    group-mean centring in this package.  What this module adds is the
    division by the within-cluster standard deviation, which is what
    separates a z-score from a centred score: centring removes the
    cluster's location, scaling also removes its spread, so clusters
    with different variability become comparable and every cluster ends
    with mean 0 and standard deviation 1 by construction.

    That last property is also the cost.  Between-cluster differences in
    both level and spread are destroyed, so a cluster-level predictor
    can no longer be recovered from ``z``; the cluster means and
    standard deviations are returned alongside so they can be
    reintroduced as level-2 predictors, which is Enders and Tofighi's
    recommendation for the centred version.

    A cluster with a single member, or with no variation, has ``sd_j =
    0`` and no defined z-score.  Those positions are returned as NaN
    rather than 0: a unit that cannot be placed relative to its cluster
    is not average, it is unmeasured.

    Parameters
    ----------
    y : array-like
        The level-1 variable.
    cluster : array-like
        Cluster identifier per observation.
    ddof : int, default 1
        Denominator correction for the within-cluster SD; 1 gives the
        sample SD, 0 the population SD.

    Returns
    -------
    RichResult
        ``estimate`` (the z vector), ``z``, ``cluster_means``,
        ``cluster_sds``, ``cluster_ids``, ``n_undefined``, ``n``.

    References
    ----------
    Raudenbush, S. W. and Bryk, A. S. (2002), Hierarchical Linear
    Models: Applications and Data Analysis Methods, 2nd ed., Sage,
    ch. 5, for within-cluster (group-mean) transformations of level-1
    predictors.  Enders, C. K. and Tofighi, D. (2007), Psychological
    Methods 12(2), 121-138, doi:10.1037/1082-989X.12.2.121, for the
    recommendation to reintroduce the cluster statistics at level 2.
    Neither source was in the local corpus; the transformation is
    arithmetic and is implemented in its standard published form,
    exactly as printed above.
    """
    v = C.vec(y)
    n = len(v)
    if n == 0:
        raise ValueError("y is empty")
    g = [str(t) for t in cluster]
    if len(g) != n:
        raise ValueError("y and cluster must have the same length")
    dd = int(ddof)
    if dd not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    base = centering_within_cluster_mean(v, g)
    cent = list(base["centered"])
    ids = list(base["cluster_ids"])
    means = list(base["cluster_means"])
    sds = []
    for cid in ids:
        idx = [i for i in range(n) if g[i] == cid]
        k = len(idx) - dd
        if k <= 0:
            sds.append(float("nan"))
            continue
        ss = 0.0
        for i in idx:
            ss += cent[i] * cent[i]
        sds.append(math.sqrt(ss / k))
    look = {cid: sds[i] for i, cid in enumerate(ids)}
    z = []
    bad = 0
    for i in range(n):
        s = look[g[i]]
        if s != s or s <= 0.0:
            z.append(float("nan"))
            bad += 1
        else:
            z.append(cent[i] / s)
    return RichResult(payload={
        "estimate": z, "z": z, "cluster_means": means, "cluster_sds": sds,
        "cluster_ids": ids, "n_undefined": bad, "n_clusters": len(ids),
        "n": n,
        "method": "Within-cluster standardisation (cluster z-score)"})


multilevelwithinclusterz = multilevel_within_cluster_z


def cheatsheet():
    return "mlwz: within-cluster z-score, z_ij = (x_ij - xbar_j) / sd_j"
