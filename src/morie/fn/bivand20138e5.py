# morie.fn -- function file (rootcoder007/morie)
"""Spatial trend surface: linear mean plus correlated error."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["spatrend", "bivand2013_chapter_8_equation_5"]


def spatrend(X, z, addintercept=True):
    """Fit the varying-mean part of a spatial model by ordinary least squares.

    A wider class than the constant-mean model is obtained by letting the
    mean vary with known spatial regressors,

        Z(s) = sum_{j=0}^{p} X_j(s) beta_j + e(s) = X beta + e(s),

    with X_0(s) identically one, so X is n by (p + 1) and beta has p + 1
    entries.  Stationarity then refers to the residual e(s) rather than
    to Z, which is why the fitted residuals are the thing to hand to a
    variogram estimator: the sample variogram of Z itself would be
    contaminated by the trend.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Spatial regressors, excluding the intercept column by default.
    z : array-like
        Observed values.
    addintercept : bool
        Prepend the column X_0(s) = 1.

    Returns
    -------
    RichResult
        ``beta``, ``fitted``, ``resid``, ``rss``, ``sigma2``, ``n``,
        ``p``.

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013),
    Applied Spatial Data Analysis with R, 2nd edn, Springer (Use R!).  Equation (8.5), p. 218: Z(s) = sum_{j=0}^p X_j(s) beta_j + e(s)
    = X beta + e(s), with the accompanying text stating that X_0(s) is
    identically one, that X is the n x (p+1) design matrix, and that for
    varying-mean models the stationarity properties refer to e(s) so the
    sample variogram must be computed from estimated residuals.  Read
    from the corpus PDF (bivand2013.pdf, p. 218).
    """
    Xm = C.mat(X)
    z = C.vec(z)
    n = len(Xm)
    if len(z) != n:
        raise ValueError("X must have one row per observation")
    if addintercept:
        Xm = C.cbind1(Xm)
    p = len(Xm[0])
    if n <= p:
        raise ValueError("need more observations than columns")
    beta, fitted, resid, xtxinv = C.lstsq(Xm, z)
    rss = sum(v * v for v in resid)
    return RichResult(payload={
        "beta": beta, "fitted": fitted, "resid": resid, "rss": rss,
        "sigma2": rss / (n - p), "n": n, "p": p,
        "method": "Spatial trend surface (Bivand et al. 2013 eq. 8.5)"})


bivand2013_chapter_8_equation_5 = spatrend


def cheatsheet():
    return "bivand20138e5: Spatial trend surface: linear mean plus correlated error."
