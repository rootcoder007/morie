# morie.fn -- function file (rootcoder007/morie)
"""BLUP of a random intercept."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["blupint", "blup_random_intercept"]


def blupint(y, group, s2u, s2e, X=None, beta=None):
    """Shrunken group effects from a random-intercept model.

    In the model y_ij = x_ij' beta + u_j + e_ij with u_j ~ N(0, s2u) and
    e_ij ~ N(0, s2e) independent, the best linear unbiased predictor of
    the group effect is the group mean residual pulled toward zero by the
    ratio of the two variance components,

        uhat_j = ( s2u / (s2u + s2e / n_j) ) * mean_j( y - x'beta ),

    a shrinkage factor that goes to one as the group grows and to zero as
    the within-group noise dominates.  With s2u = 0 every effect is
    exactly zero (complete pooling); as s2u/s2e grows without bound the
    predictor approaches the raw group mean (no pooling).

    Parameters
    ----------
    y : array-like
        Response.
    group : array-like
        Group label per observation.
    s2u : float
        Between-group variance, non-negative.
    s2e : float
        Within-group variance, strictly positive.
    X : array-like or None
        Fixed-effect design; ``None`` means an intercept-free model in
        which the raw response is used.
    beta : array-like or None
        Fixed-effect coefficients, required when ``X`` is given.

    Returns
    -------
    RichResult
        ``u``, ``shrink``, ``nj``, ``groupmean``, ``levels``, ``vpc``,
        ``J``, ``n``.

    References
    ----------
    Henderson, C. R. (1975), "Best linear unbiased estimation and
    prediction under a selection model", Biometrics 31(2), 423-447, which
    derives the mixed-model equations whose random-effect solution this
    is; Robinson, G. K. (1991), "That BLUP is a good thing: the
    estimation of random effects", Statistical Science 6(1), 15-32, gives
    the balanced one-way case in exactly the shrinkage form above.
    Standard published form; neither article was in the local corpus and
    neither was read for this implementation.
    """
    y = C.vec(y)
    n = len(y)
    g = list(group)
    if len(g) != n:
        raise ValueError("group must have one label per observation")
    s2u = float(s2u)
    s2e = float(s2e)
    if s2u < 0.0:
        raise ValueError("s2u must be non-negative")
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
    nj = []
    gm = []
    u = []
    sh = []
    for L in labs:
        idx = [i for i in range(n) if g[i] == L]
        m = len(idx)
        mean = sum(r[i] for i in idx) / m
        k = s2u / (s2u + s2e / m)
        nj.append(m)
        gm.append(mean)
        sh.append(k)
        u.append(k * mean)
    return RichResult(payload={
        "u": u, "shrink": sh, "nj": nj, "groupmean": gm,
        "levels": labs, "vpc": s2u / (s2u + s2e), "J": len(labs), "n": n,
        "method": "BLUP of a random intercept (Henderson 1975; Robinson 1991)"})


blup_random_intercept = blupint


def cheatsheet():
    return "blupr: BLUP of a random intercept."
