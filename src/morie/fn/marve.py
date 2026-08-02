# morie.fn -- function file (rootcoder007/morie)
"""Robust variance estimation for dependent effect sizes."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_robust_variance_est"]


def ma_robust_variance_est(yi, X, cluster, w=None, small_sample=True):
    r"""Robust variance estimation for meta-regression with
    DEPENDENT effect sizes (Hedges, Tipton and Johnson 2010).

    Multiple effect sizes from one study are correlated, and the
    correlation is almost never known -- different outcomes on the
    same subjects, several timepoints, overlapping samples. RVE
    sidesteps estimating it: fit weighted least squares with any
    working weights, then use the cluster-robust sandwich

    .. math:: V_R = (X'WX)^{-1}\left(\sum_g X_g'W_g e_g e_g'W_g X_g
              \right)(X'WX)^{-1},

    which is consistent for the true variance as the number of
    CLUSTERS grows, whatever the within-cluster correlation actually
    is. The working weights affect efficiency, never validity.

    The catch, and it is the one practitioners hit: the asymptotics
    are in the number of STUDIES, not effect sizes. With few
    clusters the sandwich is badly downward-biased, so Tipton's
    (2015) small-sample correction -- the HTZ adjustment with
    Satterthwaite degrees of freedom -- is applied by default, and
    ``df`` is reported. Tipton's own rule of thumb is that
    :math:`df < 4` makes the test untrustworthy regardless; that is
    flagged rather than left to be discovered.

    Parameters
    ----------
    yi : array-like, shape (m,)
        Effect sizes, possibly several per study.
    x : array-like, shape (m, p)
        Meta-regression design; a constant column is added if absent.
    cluster : array-like, shape (m,)
        Study (cluster) identifiers.
    w : array-like, optional
        Working weights; equal weights when omitted.
    small_sample : bool, default True
        Apply the HTZ small-sample correction.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``t``, ``df``, ``p``, ``vcov``,
        ``n_clusters``, ``n_effects``, ``df_warning``,
        ``small_sample``, ``method``.

    References
    ----------
    Hedges, L. V., Tipton, E. and Johnson, M. C. (2010), "Robust
    variance estimation in meta-regression with dependent effect
    size estimates", *Research Synthesis Methods* 1:39-65. Tipton,
    E. (2015), *Psychological Methods* 20:375-393, for the
    small-sample correction.
    """
    from scipy import stats

    y = np.asarray(yi, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.shape[0] != y.size:
        A = A.T
    if A.shape[0] != y.size:
        raise ValueError(f"X has {A.shape[0]} rows for {y.size} effects.")
    if not np.any(np.all(np.isclose(A, 1.0), axis=0)):
        A = np.column_stack([np.ones(y.size), A])
    g = np.asarray(cluster).ravel()
    if g.size != y.size:
        raise ValueError(f"cluster has {g.size} entries for {y.size}.")
    m, p = A.shape
    wv = np.ones(m) if w is None else np.asarray(w, dtype=float).ravel()
    if wv.size != m:
        raise ValueError(f"w has {wv.size} entries for {m} effects.")
    if np.any(wv <= 0):
        raise ValueError("working weights must be positive.")
    groups = np.unique(g)
    G = groups.size
    if G <= p:
        raise ValueError(
            f"robust variance estimation needs more clusters than "
            f"parameters: {G} studies for {p} coefficients. The asymptotics "
            "are in the number of STUDIES, not effect sizes.")
    XtWX = A.T @ (A * wv[:, None])
    bread = np.linalg.pinv(XtWX)
    beta = bread @ (A.T @ (wv * y))
    e = y - A @ beta
    meat = np.zeros((p, p))
    for gg in groups:
        s = g == gg
        Xg = A[s]
        eg = e[s]
        wg = wv[s]
        if small_sample:
            # HTZ: inflate each cluster's residual by (I - H_gg)^{-1/2}
            Hg = Xg @ bread @ (Xg * wg[:, None]).T
            I_H = np.eye(Hg.shape[0]) - Hg
            vals, vecs = np.linalg.eigh((I_H + I_H.T) / 2)
            vals = np.where(vals > 1e-10, vals ** -0.5, 0.0)
            adj = vecs @ np.diag(vals) @ vecs.T
            eg = adj @ eg
        u = Xg.T @ (wg * eg)
        meat += np.outer(u, u)
    V = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    # Satterthwaite degrees of freedom, Tipton (2015) approximation
    df = np.full(p, float(G - p))
    if small_sample:
        for j in range(p):
            num = 0.0
            den = 0.0
            for gg in groups:
                s = g == gg
                Xg = A[s]
                wg = wv[s]
                cj = (bread @ (Xg.T * wg))[j]
                q = float(cj @ cj)
                num += q
                den += q ** 2
            df[j] = (num ** 2 / den) if den > 0 else float(G - 1)
    t = np.divide(beta, se, out=np.full(p, np.nan), where=se > 0)
    pval = np.array([2 * stats.t.sf(abs(t[j]), max(df[j], 1.0))
                     for j in range(p)])
    return RichResult(payload={
        "beta": beta, "se": se, "t": t, "df": df, "p": pval, "vcov": V,
        "n_clusters": int(G), "n_effects": int(m),
        "small_sample": bool(small_sample),
        "df_warning": ("Tipton's rule of thumb: df below 4 makes the test "
                       "untrustworthy regardless of the correction"
                       if np.any(df < 4) else None),
        "asymptotics_note": "consistent as the number of STUDIES grows, "
                            "whatever the within-study correlation is; the "
                            "working weights affect efficiency, never "
                            "validity",
        "method": "Robust variance estimation for dependent effects "
                  "(Hedges-Tipton-Johnson 2010; Tipton 2015 correction)"})


def cheatsheet():
    return "marve: clusters are studies, not effects -- and df < 4 is untrustworthy whatever the correction"
