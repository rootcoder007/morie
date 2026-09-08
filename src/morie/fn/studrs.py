# morie.fn -- function file (rootcoder007/morie)
"""Externally studentized residuals."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["studentized_residual"]


def studentized_residual(y, X):
    """Residuals rescaled by a variance the point itself did not inflate.

    An outlier drags the fitted line towards itself and inflates the
    residual standard error, so the ordinary standardised residual
    understates how odd the point is -- twice over.  Deleting the point
    from the scale estimate breaks that circularity, which is why the
    externally studentized version is the one that follows a t
    distribution.

    Formula: ``t_i = e_i / (s_(i) sqrt(1 - h_ii))`` with
    ``s_(i)^2 = [(n - p) s^2 - e_i^2 / (1 - h_ii)] / (n - p - 1)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design; supply your own intercept column.

    Returns
    -------
    RichResult
        ``estimate`` (the largest absolute value), ``t``, ``leverage``,
        ``sigma``, ``df``, ``n``.

    References
    ----------
    Weisberg, S. (2014).  Applied Linear Regression, 4th edition.
    Wiley, section 9.1.  The deletion form is Belsley, D. A., Kuh, E. &
    Welsch, R. E. (1980), Regression Diagnostics, Wiley, chapter 2.
    """
    Xm = C.mat(X)
    yv = C.vec(y)
    n, p = C.shape(Xm)
    beta, fitted, resid, xtxinv = S.ols(Xm, yv)
    h = [C.dot(Xm[i], C.matvec(xtxinv, Xm[i])) for i in range(n)]
    s2 = sum(t * t for t in resid) / (n - p)
    t = []
    for i in range(n):
        si2 = ((n - p) * s2 - resid[i] ** 2 / (1.0 - h[i])) / (n - p - 1)
        den = math.sqrt(si2 * (1.0 - h[i])) if si2 > 0 else float("nan")
        t.append(resid[i] / den if den == den and den > 0 else float("nan"))
    big = max(range(n), key=lambda i: abs(t[i]) if t[i] == t[i] else -1.0)
    return RichResult(payload={
        "estimate": t[big], "t": t, "leverage": h, "sigma": math.sqrt(s2),
        "df": n - p - 1, "n": n,
        "method": "Externally studentized residuals"})


def cheatsheet():
    return "studrs: Externally studentized residuals."
