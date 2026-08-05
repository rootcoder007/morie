# morie.fn -- function file (rootcoder007/morie)
"""DFFITS: scaled change in the fitted value when observation i is deleted."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dffits"]


def _ols_influence(y, X, intercept=True):
    """Hat values, residuals and (X'X)^-1 for the OLS fit of y on X.

    Returns (n, p, e, h, sse, inv) with ``inv[k]`` the k-th COLUMN of
    (X'X)^-1.  The design gets a leading intercept column unless the
    caller turns it off.
    """
    y = core.vec(y)
    Xm = core.mat(X)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(Xm) != n:
        raise ValueError("y and X must have the same number of rows")
    D = [([1.0] + list(r)) if intercept else [float(v) for v in r] for r in Xm]
    p = len(D[0])
    if n <= p + 1:
        raise ValueError("need n > p + 1 observations for deletion diagnostics")
    XtX = core.crossprod(D)
    inv = [core.cholsolve(XtX, [1.0 if j == k else 0.0 for j in range(p)])
           for k in range(p)]
    beta = core.cholsolve(XtX, core.matvec(core.tr(D), y))
    fit = core.matvec(D, beta)
    e = [y[i] - fit[i] for i in range(n)]
    h = []
    for i in range(n):
        s = 0.0
        for a in range(p):
            for b in range(p):
                s += D[i][a] * inv[b][a] * D[i][b]
        h.append(s)
    sse = sum(v * v for v in e)
    return n, p, e, h, sse, inv


def _s_deleted(sse, e_i, h_i, n, p):
    """s_(i), the residual sd with observation i removed."""
    num = sse - e_i * e_i / (1.0 - h_i)
    return math.sqrt(max(num, 0.0) / (n - p - 1))


def dffits(y, X, intercept=True):
    """
    DFFITS scaled change in fitted value when obs i deleted

    Formula: DFFITS_i = (e_i / (s_(i) sqrt(1 - h_ii))) sqrt(h_ii / (1 - h_ii))

    The cut-off is the Belsley-Kuh-Welsch size-adjusted 2 sqrt(p/n).

    Parameters
    ----------
    y : array-like
        Response vector, length n.
    X : array-like
        Design matrix, n rows.  An intercept column is prepended unless
        ``intercept`` is False.
    intercept : bool
        Whether to prepend a column of ones.

    Returns
    -------
    result : dict
        Keys: estimate (max |DFFITS|), dffits, threshold, flagged,
        n_influential, n, p.

    References
    ----------
    Belsley, Kuh, Welsch (1980), Regression Diagnostics, Wiley, ch. 2.
    """
    n, p, e, h, sse, _inv = _ols_influence(y, X, intercept)
    out = []
    for i in range(n):
        si = _s_deleted(sse, e[i], h[i], n, p)
        if si <= 0.0 or h[i] >= 1.0:
            out.append(float("nan"))
        else:
            out.append(e[i] * math.sqrt(h[i]) / (si * (1.0 - h[i])))
    thr = 2.0 * math.sqrt(float(p) / n)
    flagged = [1 if (v == v and abs(v) > thr) else 0 for v in out]
    finite = [abs(v) for v in out if v == v]
    return RichResult(payload={
        "estimate": max(finite) if finite else float("nan"),
        "dffits": out,
        "threshold": thr,
        "flagged": flagged,
        "n_influential": sum(flagged),
        "n": n,
        "p": p,
        "method": "DFFITS scaled change in fitted value when obs i deleted",
    })


def cheatsheet():
    return "dffit: DFFITS scaled change in fitted value when obs i deleted"
