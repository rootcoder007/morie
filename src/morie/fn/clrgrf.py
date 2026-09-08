# morie.fn -- function file (rootcoder007/morie)
r"""Cluster-aware generalized random forests.

When observations arrive in clusters -- pupils in schools, patients in
clinics, repeat visits by one person -- the independent unit is the
cluster, not the row. A forest that ignores this is not merely
conservative in the wrong direction; it is wrong twice over.

**The subsample has to draw clusters.** If rows are drawn
independently, most clusters land partly in the sample used to place a
split and partly in the sample used to estimate the leaf. Honesty is
then violated *through the cluster*: the leaf average is informed by
rows correlated with the ones that chose the split. Drawing whole
clusters restores it.

**The variance has to aggregate to the cluster.** The infinitesimal
jackknife is computed against the cluster indicator :math:`N^*_{cb}`
rather than the row indicator, so what is measured is the effect of
adding or removing a whole cluster. With :math:`m` clusters the
correction becomes :math:`m(m-1)/(m - s_c)^2` in the number of
*clusters*, not rows.

Both halves matter and they fail differently: keeping row-level
subsampling but cluster-level variance leaves the honesty violation in
place while the interval looks respectable. The anchor measures the
interval width under strong within-cluster correlation, where the
row-level version is visibly too narrow.

**Equal weight per cluster, not per row.** Clusters differ in size, so
predictions weight each cluster equally by default (``unit="cluster"``)
rather than letting a large cluster count more than a small one; the
row-weighted alternative is exposed because both estimands are
legitimate and they answer different questions.

References
----------
Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. Eq. (2)-(3); the grf software
that accompanies it implements clustered sampling in this form.

Wager, S. & Athey, S. (2018) "Estimation and Inference of Heterogeneous
Treatment Effects using Random Forests", *Journal of the American
Statistical Association* 113(523), 1228-1242,
doi:10.1080/01621459.2017.1319839. Eq. (8), the infinitesimal jackknife
this aggregates.

Cameron, A. C. & Miller, D. L. (2015) "A Practitioner's Guide to
Cluster-Robust Inference", *Journal of Human Resources* 50(2), 317-372,
doi:10.3368/jhr.50.2.317. Why the cluster is the unit and what ignoring
it costs.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest, leaf_of

__all__ = ["cluster_forest", "cluster_jackknife", "cluster_index"]

_EPS = 1e-12


def cluster_index(clusters):
    """Group row indices by cluster label, in first-seen order."""
    lab = [str(c) for c in clusters]
    order, groups = [], {}
    for i, c in enumerate(lab):
        if c not in groups:
            groups[c] = []
            order.append(c)
        groups[c].append(i)
    return [groups[c] for c in order], order


def cluster_jackknife(preds, bags, groups, correction=True):
    r"""Eq. (8) aggregated to the cluster.

    The covariance is taken against whether the whole CLUSTER was in
    tree b, and the finite-sample factor counts clusters.
    """
    B = len(preds)
    m = len(groups)
    if B < 2:
        raise ValueError("clrgrf: need at least 2 trees, got %d" % B)
    if m < 3:
        raise ValueError("clrgrf: need at least 3 clusters, got %d" % m)
    # a cluster is in-bag for tree b if any of its rows is
    inbag_c = [[any(bags[b][i] for i in g) for g in groups]
               for b in range(B)]
    # the AVERAGE number of clusters in a tree, not tree 0's count
    sc = sum(sum(1 for c in range(m) if inbag_c[b][c])
             for b in range(B)) / float(B)
    subsampled = sc < m - 0.5
    pbar = sum(preds) / B
    total = 0.0
    for c in range(m):
        nbar = sum(1.0 if inbag_c[b][c] else 0.0
                   for b in range(B)) / B
        cov = sum((preds[b] - pbar)
                  * ((1.0 if inbag_c[b][c] else 0.0) - nbar)
                  for b in range(B)) / B
        total += cov * cov
    # With ROW-level sampling nearly every cluster has some row in every
    # tree, so no cluster is ever left out and the cluster-level
    # correction is undefined. That is not an error -- it is the
    # diagnostic: row sampling does not subsample clusters at all, so
    # the interval it produces cannot reflect cluster-level uncertainty.
    if correction and subsampled:
        total *= (m - 1.0) / m * (float(m) / (m - sc)) ** 2
    return total, {"clusters_per_tree": sc, "subsampled": subsampled,
                   "m": m}


def cluster_forest(y, X, clusters, at=None, n_trees=200, min_leaf=5,
                   subsample_frac=0.5, seed=0, unit="cluster",
                   level=0.95, cluster_sampling=True):
    r"""A forest whose independent unit is the cluster.

    ``cluster_sampling=False`` reverts to row-level subsampling while
    keeping the cluster-level variance, which is the half-measure worth
    being able to measure: honesty is violated through the cluster but
    the interval still looks respectable.
    """
    if unit not in ("cluster", "row"):
        raise ValueError("clrgrf: unit must be cluster or row, got %r"
                         % (unit,))
    yv = k.vec(y)
    n = len(yv)
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("clrgrf: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    if len(clusters) != n:
        raise ValueError("clrgrf: %d cluster labels for %d rows"
                         % (len(clusters), n))
    groups, labels = cluster_index(clusters)
    m = len(groups)
    if m < 6:
        raise ValueError("clrgrf: need at least 6 clusters, got %d" % m)

    trees, bags, s = grow_forest(
        Xm, yv, n_trees=n_trees, min_leaf=min_leaf,
        subsample_frac=subsample_frac, seed=seed,
        clusters=(clusters if cluster_sampling else None))

    Q = k.mat(at) if at is not None else Xm
    fitted, var = [], []
    for q in range(len(Q)):
        per_tree = []
        for tree in trees:
            node, _ = leaf_of(tree, Q[q])
            rows = node["I"]
            if not rows:
                per_tree.append(0.0)
                continue
            if unit == "row":
                per_tree.append(sum(yv[i] for i in rows) / len(rows))
            else:
                # equal weight per cluster present in the leaf
                byc = {}
                for i in rows:
                    byc.setdefault(str(clusters[i]), []).append(yv[i])
                per_tree.append(sum(sum(v) / len(v)
                                    for v in byc.values()) / len(byc))
        fitted.append(sum(per_tree) / len(per_tree))
        v, vinfo = cluster_jackknife(per_tree, bags, groups)
        var.append(v)
    se = [math.sqrt(max(v, 0.0)) for v in var]
    z = k.qnorm(0.5 + 0.5 * float(level))
    return RichResult(payload={
        "estimate": fitted, "fitted": fitted, "se": se,
        "ci": [(fitted[q] - z * se[q], fitted[q] + z * se[q])
               for q in range(len(Q))],
        "variance": var, "n": n, "n_clusters": m,
        "clusters_per_tree": vinfo["clusters_per_tree"],
        "clusters_subsampled": vinfo["subsampled"],
        "cluster_sizes": [len(g) for g in groups],
        "cluster_labels": labels, "unit": unit,
        "cluster_sampling": bool(cluster_sampling),
        "n_trees": int(n_trees), "level": float(level),
        "method": "cluster-aware generalized random forest, Athey, "
                  "Tibshirani & Wager (2019) with eq. (8) aggregated "
                  "to the cluster",
    })


def cheatsheet():
    return ("clrgrf: draw whole CLUSTERS into the subsample -- row-wise "
            "draws split clusters across the split and estimate halves, "
            "violating honesty through the cluster -- and take the IJ "
            "covariance against the cluster indicator with the "
            "m(m-1)/(m-sc)^2 factor counting clusters.")


# compact alias per ledger/NAMING.md
clusterforest = cluster_forest

# public names resolved by fn/_lazy_map.json
clustered_grf = cluster_forest
clusteredgrf = cluster_forest
