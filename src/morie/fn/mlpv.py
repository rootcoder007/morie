# morie.fn -- function file (rootcoder007/morie)
"""Proportional reduction in level-1 variance across nested models."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["multilevel_pseudo_variance_ratio", "multilevelpseudovarianceratio"]


def multilevel_pseudo_variance_ratio(y, X, cluster):
    """Pseudo-R^2 at level 1: how much residual variance the predictors remove.

        PR = (sigma2_e(null) - sigma2_e(full)) / sigma2_e(null)

    The null model is the one-way random-effects ANOVA -- cluster
    intercepts and nothing else -- and the full model adds the level-1
    predictors ``X`` on top of those same intercepts.  Both residual
    variances are computed on the within-cluster (fixed-effects)
    transformation, which sweeps the cluster means out of ``y`` and
    ``X``; that is what makes them comparable, since both are then
    variances of the same deviations about the same intercepts.

    Degrees of freedom are ``n - J`` for the null model and
    ``n - J - p`` for the full one, ``J`` being the number of clusters.
    Using them, rather than dividing both by ``n``, is what stops a
    predictor from appearing to explain variance purely by being
    counted.

    ``PR`` is NOT bounded below by zero.  A negative value is a real and
    informative outcome -- it says the added predictors cost more
    degrees of freedom than they repaid -- and it is reported as it
    comes out rather than clamped, which is exactly the diagnostic
    Raudenbush and Bryk warn is lost when the quantity is presented as
    an R-squared.

    Parameters
    ----------
    y : array-like, length n
        Response.
    X : array-like, shape (n, p)
        Level-1 predictors, without an intercept column; a flat
        length-n sequence is read as a single predictor.
    cluster : array-like, length n
        Cluster identifier per observation.

    Returns
    -------
    RichResult
        ``estimate`` (PR), ``pr``, ``sigma2_null``, ``sigma2_full``,
        ``df_null``, ``df_full``, ``n_clusters``, ``p``, ``n``.

    References
    ----------
    Raudenbush, S. W. and Bryk, A. S. (2002), Hierarchical Linear
    Models: Applications and Data Analysis Methods, 2nd ed., Sage,
    ch. 4, "proportion reduction in variance" at level 1.  The book was
    not in the local corpus and could not be obtained; the quantity is
    implemented exactly as the ratio printed above, which is its
    standard published form, and it is stated here in full so it can be
    checked against the chapter by anyone who has it.
    """
    v = C.vec(y)
    n = len(v)
    if n == 0:
        raise ValueError("y is empty")
    g = [str(t) for t in cluster]
    if len(g) != n:
        raise ValueError("y and cluster must have the same length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("X must have one row per observation")
    p = len(Xm[0])
    for row in Xm:
        if len(row) != p:
            raise ValueError("X rows must all have the same length")
    ids = []
    for cid in g:
        if cid not in ids:
            ids.append(cid)
    J = len(ids)
    if n - J - p <= 0:
        raise ValueError("not enough observations: n - J - p must be positive")

    # Within transformation: subtract the cluster mean from y and from
    # every column of X.  Cluster intercepts vanish, so what is left is
    # exactly the level-1 residual structure both models share.
    def sweep(col):
        out = list(col)
        for cid in ids:
            idx = [i for i in range(n) if g[i] == cid]
            mu = sum(col[i] for i in idx) / len(idx)
            for i in idx:
                out[i] = col[i] - mu
        return out

    yw = sweep(v)
    ssw = sum(t * t for t in yw)
    s2_null = ssw / (n - J)
    if s2_null <= 0.0:
        raise ValueError("the null model has no within-cluster variance; "
                         "the ratio is undefined")
    Xw = [sweep([Xm[i][j] for i in range(n)]) for j in range(p)]
    dm = [[Xw[j][i] for j in range(p)] for i in range(n)]
    resid = C.lstsq(dm, yw)[2]
    s2_full = sum(t * t for t in resid) / (n - J - p)
    pr = (s2_null - s2_full) / s2_null
    return RichResult(payload={
        "estimate": pr, "pr": pr, "sigma2_null": s2_null,
        "sigma2_full": s2_full, "df_null": n - J, "df_full": n - J - p,
        "n_clusters": J, "p": p, "n": n,
        "method": "Proportional reduction in level-1 variance"})


multilevelpseudovarianceratio = multilevel_pseudo_variance_ratio


def cheatsheet():
    return "mlpv: proportional reduction in level-1 residual variance"
