# morie.fn -- function file (rootcoder007/morie)
"""BLUP of random intercept and slope."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["blupslope", "blup_random_slope"]


def blupslope(y, group, Z, D, s2e, X=None, beta=None):
    """Group-level random-coefficient vectors from the mixed-model solution.

    With y_j = X_j beta + Z_j v_j + e_j, v_j ~ N(0, D) and
    e_j ~ N(0, s2e I) independent across groups, the best linear unbiased
    predictor of the group's random-coefficient vector is

        vhat_j = D Z_j' ( Z_j D Z_j' + s2e I )^{-1} ( y_j - X_j beta ),

    which is the random-intercept shrinkage generalised to a vector: the
    prior covariance D scales the effect up, the residual covariance
    scales it down, and the inverse in the middle is the marginal
    covariance of that group's observations.  The equivalent smaller
    solve, vhat_j = (Z_j'Z_j/s2e + D^{-1})^{-1} Z_j'(y_j - X_j beta)/s2e,
    is algebraically identical; the form above is used because it stays
    defined when D is singular.

    Parameters
    ----------
    y : array-like
        Response.
    group : array-like
        Group label per observation.
    Z : array-like, shape (n, q)
        Random-effect design, e.g. a column of ones and one covariate.
    D : array-like, shape (q, q)
        Covariance of the random coefficients.
    s2e : float
        Residual variance, strictly positive.
    X : array-like or None
        Fixed-effect design.
    beta : array-like or None
        Fixed-effect coefficients, required when ``X`` is given.

    Returns
    -------
    RichResult
        ``v``, ``levels``, ``nj``, ``fitted``, ``J``, ``q``, ``n``.

    References
    ----------
    Henderson, C. R. (1975), "Best linear unbiased estimation and
    prediction under a selection model", Biometrics 31(2), 423-447, whose
    mixed-model equations give vhat = D Z' V^{-1} (y - X beta) with
    V = Z D Z' + R; Robinson, G. K. (1991), Statistical Science 6(1),
    15-32, is the standard exposition.  Standard published form; neither
    article was in the local corpus and neither was read for this
    implementation.
    """
    y = C.vec(y)
    n = len(y)
    g = list(group)
    if len(g) != n:
        raise ValueError("group must have one label per observation")
    Zm = C.mat(Z)
    if len(Zm) != n:
        raise ValueError("Z must have one row per observation")
    q = len(Zm[0])
    Dm = C.mat(D)
    if len(Dm) != q or len(Dm[0]) != q:
        raise ValueError("D must be q by q")
    s2e = float(s2e)
    if s2e <= 0.0:
        raise ValueError("s2e must be strictly positive")
    if X is None:
        r = list(y)
    else:
        Xm = C.mat(X)
        if len(Xm) != n:
            raise ValueError("X must have one row per observation")
        b = C.vec(beta)
        if len(b) != len(Xm[0]):
            raise ValueError("beta must have one entry per column of X")
        r = [y[i] - sum(Xm[i][j] * b[j] for j in range(len(b)))
             for i in range(n)]
    labs = []
    for v in g:
        if v not in labs:
            labs.append(v)
    V = []
    nj = []
    fit = [0.0] * n
    for L in labs:
        idx = [i for i in range(n) if g[i] == L]
        m = len(idx)
        nj.append(m)
        Zj = [Zm[i] for i in idx]
        ZD = C.matmul(Zj, Dm)
        M = [[sum(ZD[a][t] * Zj[b][t] for t in range(q))
              + (s2e if a == b else 0.0) for b in range(m)]
             for a in range(m)]
        w = C.solvev(M, [r[i] for i in idx])
        DZt = [[sum(Dm[a][t] * Zj[b][t] for t in range(q))
                for b in range(m)] for a in range(q)]
        vj = [sum(DZt[a][b] * w[b] for b in range(m)) for a in range(q)]
        V.append(vj)
        for i in idx:
            fit[i] = sum(Zm[i][a] * vj[a] for a in range(q))
    return RichResult(payload={
        "v": V, "levels": labs, "nj": nj, "fitted": fit,
        "J": len(labs), "q": q, "n": n,
        "method": "BLUP of random coefficients (Henderson 1975; Robinson 1991)"})


blup_random_slope = blupslope


def cheatsheet():
    return "blups: BLUP of random intercept and slope."
