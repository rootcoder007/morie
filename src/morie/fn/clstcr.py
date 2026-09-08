# morie.fn -- function file (rootcoder007/morie)
"""Cluster-level causal inference with cluster-robust inference."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["cluster_causal_inference"]


def cluster_causal_inference(y, D, cluster, X=None, alpha=0.05):
    r"""Treatment effect with cluster-randomised assignment.

    When treatment is assigned at the cluster level, unit-level
    standard errors are anticonservative. This fits the unit-level
    regression :math:`Y = \beta_0 + \tau D + \beta' X` and reports the
    cluster-robust (CR0 sandwich) variance

    .. math:: \hat V = (X'X)^{-1}
              \Big(\sum_g X_g' u_g u_g' X_g\Big) (X'X)^{-1},

    with the small-sample correction :math:`G/(G-1)` and a
    :math:`t_{G-1}` reference distribution. It also reports the
    cluster-mean ("collapsed") estimate, which is exactly valid under
    cluster randomisation with equal cluster sizes and is the honest
    fallback when G is small.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment (constant within cluster if truly cluster-assigned).
    cluster : array-like, shape (n,)
        Cluster identifier.
    X : array-like, optional
        Unit-level covariates.
    alpha : float, default 0.05
        Two-sided level.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``se_cluster``, ``se_naive``, ``ci``,
        ``p_value``, ``estimate_collapsed`` (difference of cluster
        means), ``n_clusters``, ``icc`` (one-way ANOVA estimate),
        ``n``, ``method``.

    References
    ----------
    Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
    cluster-robust inference. *Journal of Human Resources*, 50(2),
    317-372.
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    g = np.asarray(cluster).ravel()
    n = y.size
    if not (D.size == n and g.size == n):
        raise ValueError("y, D, cluster must have equal length.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    groups, inv = np.unique(g, return_inverse=True)
    G = groups.size
    if G < 3:
        raise ValueError(f"need at least 3 clusters, got {G}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    cols = [np.ones(n), D]
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        if Xa.shape[0] != n:
            raise ValueError(f"X has {Xa.shape[0]} rows but y has {n}.")
        cols.append(Xa)
    Dm = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Dm, y, rcond=None)
    u = y - Dm @ b
    XtX_inv = np.linalg.pinv(Dm.T @ Dm)

    meat = np.zeros((Dm.shape[1], Dm.shape[1]))
    for j in range(G):
        s = inv == j
        Xg, ug = Dm[s], u[s]
        sc = Xg.T @ ug
        meat += np.outer(sc, sc)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se_c = float(np.sqrt(V[1, 1]))

    s2 = float((u**2).sum() / max(n - Dm.shape[1], 1))
    se_n = float(np.sqrt(s2 * XtX_inv[1, 1]))

    tcrit = stats.t.ppf(1 - alpha / 2, G - 1)
    est = float(b[1])
    p = float(2 * stats.t.sf(abs(est / se_c), G - 1)) if se_c > 0 else float("nan")

    cm_y = np.array([y[inv == j].mean() for j in range(G)])
    cm_d = np.array([D[inv == j].mean() for j in range(G)])
    treated = cm_d > 0.5
    collapsed = (
        float(cm_y[treated].mean() - cm_y[~treated].mean())
        if treated.any() and (~treated).any()
        else float("nan")
    )

    # one-way ANOVA intraclass correlation of the outcome
    grand = y.mean()
    sizes = np.array([(inv == j).sum() for j in range(G)])
    msb = float((sizes * (cm_y - grand) ** 2).sum() / (G - 1))
    msw = float(sum(((y[inv == j] - cm_y[j]) ** 2).sum() for j in range(G)) / max(n - G, 1))
    nbar = float(sizes.mean())
    icc = (msb - msw) / (msb + (nbar - 1) * msw) if (msb + (nbar - 1) * msw) > 0 else float("nan")

    return RichResult(
        payload={
            "estimate": est,
            "se_cluster": se_c,
            "se_naive": se_n,
            "ci": (est - tcrit * se_c, est + tcrit * se_c),
            "p_value": p,
            "estimate_collapsed": collapsed,
            "n_clusters": int(G),
            "icc": float(icc),
            "n": int(n),
            "method": "Cluster-level causal effect with CR0 cluster-robust inference",
        }
    )


def cheatsheet():
    return "clstcr: CR0 sandwich SE with G/(G-1) and t_{G-1}; naive SE reported for contrast"
