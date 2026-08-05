# morie.fn -- function file (rootcoder007/morie)
"""Cluster-robust inference for the doubly robust DiD estimator.

The point estimate is the doubly robust moment of Sant'Anna, P. H. C.
and Zhao, J. (2020), *Journal of Econometrics* 219(1), 101-122, eq.
(2.6), which the shared core already supplies together with its
influence function psi_i.  The contribution here is the variance.

Bertrand, M., Duflo, E. and Mullainathan, S. (2004), *Quarterly Journal
of Economics* 119(1), 249-275, show that treating serially correlated
panel observations as independent understates the DiD standard error by
a large factor, and that clustering on the unit of treatment assignment
fixes it.  So the influence function is summed within cluster before it
is squared,

    V_CR = (1 / n^2) sum_c ( sum_{i in c} psi_i )^2 * a,
    a    = [G / (G - 1)] * [(n - 1) / (n - k)],

the finite-cluster correction carried by Stata's ``vce(cluster)`` and by
the sandwich literature; k is the number of columns of the outcome
design.  With one observation per cluster the sum collapses and V_CR is
the independent variance up to a, which is the degenerate check.

Also reported is the variance-inflation factor V_CR / V_iid: it is the
quantity Bertrand et al. argue is routinely much larger than one.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_clustered_did"]


def dr_clustered_did(y, D, X=None, cluster=None):
    """DR-DiD point estimate with a cluster-robust standard error.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates.
    cluster : array-like, optional
        Cluster label per unit; ``None`` puts every unit in its own
        cluster, which reduces to the independent variance times ``a``.

    Returns
    -------
    result : dict
        Keys: estimate, se, se_iid, vif, n_clusters, dof_adj, k, n.

    References
    ----------
    Bertrand, Duflo & Mullainathan (2004), QJE 119(1):249-275,
    doi:10.1162/003355304772839588.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6),
    doi:10.1016/j.jeconom.2020.06.003.
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    cl = [str(x) for x in cluster] if cluster is not None else [str(i) for i in range(n)]
    if len(cl) != n:
        raise ValueError("cluster must have the same length as y")
    fit = k.drdid_panel(yv, dv, X)
    psi = fit["inf"]
    labels = []
    for c in cl:
        if c not in labels:
            labels.append(c)
    G = len(labels)
    nk = 1 + (k.ncol(k.mat(X)) if X is not None else 0)
    if G < 2 or n <= nk:
        adj = 1.0
    else:
        adj = (G / (G - 1.0)) * ((n - 1.0) / (n - nk))
    v = 0.0
    for c in labels:
        s = 0.0
        for i in range(n):
            if cl[i] == c:
                s += psi[i]
        v += s * s
    v_cr = adj * v / (n * n)
    v_iid = 0.0
    for x in psi:
        v_iid += x * x
    v_iid = v_iid / (n * n)
    return RichResult(
        title="Cluster-robust DR-DiD",
        summary_lines=[("clusters", G)],
        payload={
            "estimate": fit["tau"],
            "se": v_cr ** 0.5,
            "se_iid": v_iid ** 0.5,
            "vif": (v_cr / v_iid) if v_iid > 0.0 else float("nan"),
            "n_clusters": G,
            "dof_adj": adj,
            "k": nk,
            "n": n,
            "method": "Cluster-robust DR-DiD",
        },
    )


def cheatsheet():
    return "drclt: Cluster-robust DR-DiD"


# compact alias per ledger/NAMING.md
drclustereddid = dr_clustered_did
