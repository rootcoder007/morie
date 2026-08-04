# morie.fn -- function file (rootcoder007/morie)
"""DFFITS, the scaled deletion influence on the fitted value."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dffitsols']


def dffitsols(X, y, intercept=True):
    """DFFITS, the scaled deletion influence on the fitted value.

    One number per observation combining how unusual its predictors are (leverage) with how badly the model fits it (the externally studentised residual). Note a naming hazard: the public name ``dffits`` is already registered to the separate module morie.fn.dffit, so this implementation is registered as ``dffitsols`` and the duplicate is left for the maintainers to resolve rather than silently shadowed.


    Formula: DFFITS_i = t*_i sqrt(h_ii/(1-h_ii)), t*_i = e_i / (s_(i) sqrt(1-h_ii))

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    intercept : bool
        Prepend a column of ones.

    Returns
    -------
    RichResult
        ``dffits``, ``cutoff``, ``leverage``, ``student``, ``n``, ``p``.

    References
    ----------
    Belsley, Kuh and Welsch (1980), Regression Diagnostics: Identifying
    Influential Data and Sources of Collinearity, Wiley.  The book is
    not held locally; the definitions and the cutoffs used here
    (2/sqrt(n) for DFBETAS, 2 sqrt(p/n) for DFFITS, scaling by the
    delete-one root mean square s_(i)) are as documented by the SAS
    and R reference implementations, which cite BKW for them.
    """
    X = C.mat(X)
    if intercept:
        X = C.cbind1(X)
    y = C.vec(y)
    n = len(X); p = len(X[0])
    beta, fit, res, xtxinv = C.lstsq(X, y)
    h = C.hatdiag(X, xtxinv)
    rss = sum(v * v for v in res)
    df = n - p
    dff, stu = [], []
    for i in range(n):
        d = 1.0 - h[i]
        if d <= 0 or df <= 1:
            dff.append(float("nan")); stu.append(float("nan")); continue
        s2i = (rss - res[i] * res[i] / d) / (df - 1)
        s = math.sqrt(s2i) if s2i > 0 else float("nan")
        t = res[i] / (s * math.sqrt(d))
        stu.append(t)
        dff.append(t * math.sqrt(h[i] / d))
    return RichResult(payload={
        "dffits": dff, "cutoff": 2.0 * math.sqrt(p / float(n)),
        "leverage": h, "student": stu, "n": n, "p": p,
        "method": "DFFITS (Belsley-Kuh-Welsch)"})



def cheatsheet():
    return "dffits: DFFITS, the scaled deletion influence on the fitted value."
