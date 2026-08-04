# morie.fn -- slice s03 (rootcoder007/morie)
"""Centering within cluster mean.

Source consulted: Enders, C. K. and Tofighi, D. (2007).  Centering
predictor variables in cross-sectional multilevel models: a new look at
an old issue.  *Psychological Methods* 12(2), 121-138.  Centering within
cluster (CWC, also group-mean centering) is

    x_ij(CWC) = x_ij - xbar_j

with xbar_j the mean of cluster j.  The paper is paywalled; the
transformation is arithmetic and is quoted in its standard published
form.  CWC removes all between-cluster variance from the predictor, so
the cluster means are returned as well -- Enders and Tofighi's
recommendation is to reintroduce them as a level-2 predictor rather
than discard them.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["centering_within_cluster_mean"]


def centering_within_cluster_mean(y, cluster):
    """Centre a level-1 covariate within each cluster.

    Parameters
    ----------
    y : array-like
        The covariate.
    cluster : array-like
        Cluster identifier per observation, same length as ``y``.

    Returns
    -------
    RichResult with payload:
        estimate      : the within-cluster centred vector
        centered      : same as estimate
        cluster_means : mean per cluster, in order of first appearance
        cluster_ids   : the cluster labels in that same order
        icc_between   : between-cluster share of the total sum of squares
    """
    v = k.vec(y)
    g = [str(c) for c in cluster]
    ids = []
    for c in g:
        if c not in ids:
            ids.append(c)
    means = []
    for c in ids:
        sub = [v[i] for i in range(len(v)) if g[i] == c]
        means.append(k.mean(sub))
    lookup = {c: means[i] for i, c in enumerate(ids)}
    cent = [v[i] - lookup[g[i]] for i in range(len(v))]
    gm = k.mean(v)
    ssb = 0.0
    for i, c in enumerate(ids):
        nj = 0
        for x in g:
            if x == c:
                nj += 1
        ssb += nj * (means[i] - gm) ** 2
    sst = 0.0
    for x in v:
        sst += (x - gm) ** 2
    return RichResult(
        title="Centering within cluster (CWC)",
        summary_lines=[("clusters", len(ids))],
        payload={
            "estimate": cent,
            "centered": cent,
            "cluster_means": means,
            "cluster_ids": ids,
            "icc_between": ssb / sst if sst > 0.0 else float("nan"),
            "n": len(v),
            "method": "Centering within cluster mean (CWC)",
        },
    )


def cheatsheet():
    return "cwcm: Centering within cluster mean (CWC) for level-1 covariate"
