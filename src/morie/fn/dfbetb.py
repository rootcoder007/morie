# morie.fn -- function file (rootcoder007/morie)
"""DFBETAS: scaled change in each coefficient when observation i is deleted."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .dffit import _ols_influence, _s_deleted

__all__ = ["dfbetas"]


def dfbetas(y, X, intercept=True):
    """
    DFBETAS scaled change in coefficient j when obs i deleted

    Formula: DFBETAS_ij = (beta_j - beta_j(-i)) / (s_(i) sqrt((X'X)^-1_jj))

    The deletion update beta - beta(-i) = (X'X)^-1 x_i e_i / (1 - h_ii)
    avoids refitting n times.  Cut-off 2/sqrt(n).

    Parameters
    ----------
    y : array-like
        Response vector, length n.
    X : array-like
        Design matrix, n rows.
    intercept : bool
        Whether to prepend a column of ones.

    Returns
    -------
    result : dict
        Keys: estimate (max |DFBETAS|), dfbetas (n x p), threshold,
        n_influential, n, p.

    References
    ----------
    Belsley, Kuh, Welsch (1980), Regression Diagnostics, Wiley, ch. 2.
    """
    n, p, e, h, sse, inv = _ols_influence(y, X, intercept)
    Xm = core.mat(X)
    D = [([1.0] + list(r)) if intercept else [float(v) for v in r] for r in Xm]
    out = []
    worst = 0.0
    n_infl = 0
    thr = 2.0 / math.sqrt(float(n))
    for i in range(n):
        si = _s_deleted(sse, e[i], h[i], n, p)
        row = []
        flag = 0
        for j in range(p):
            num = 0.0
            for a in range(p):
                num += inv[j][a] * D[i][a]
            denom = si * math.sqrt(inv[j][j]) * (1.0 - h[i])
            v = num * e[i] / denom if denom != 0.0 else float("nan")
            row.append(v)
            if v == v:
                worst = max(worst, abs(v))
                if abs(v) > thr:
                    flag = 1
        n_infl += flag
        out.append(row)
    return RichResult(payload={
        "estimate": worst,
        "dfbetas": out,
        "threshold": thr,
        "n_influential": n_infl,
        "n": n,
        "p": p,
        "method": "DFBETAS scaled change in coefficient when obs i deleted",
    })


def cheatsheet():
    return "dfbetb: DFBETAS scaled change in coefficient j when obs i deleted"
