# morie.fn -- function file (rootcoder007/morie)
"""COVRATIO: effect of deleting observation i on the covariance of beta-hat."""

import math

from ._richresult import RichResult
from .dffit import _ols_influence, _s_deleted

__all__ = ["covratio"]


def covratio(y, X, intercept=True):
    """
    COVRATIO effect of deleting obs i on the precision of beta-hat

    Formula: COVRATIO_i = (s_(i)^2 / s^2)^p / (1 - h_ii)

    It is the ratio of generalised variances
    det(s_(i)^2 (X_(i)'X_(i))^-1) / det(s^2 (X'X)^-1).  Values far from 1
    mark observations that change the precision; the Belsley-Kuh-Welsch
    cut-off is |COVRATIO - 1| > 3p/n.

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
        Keys: estimate (max |COVRATIO - 1|), covratio, threshold,
        flagged, n_influential, n, p.

    References
    ----------
    Belsley, Kuh, Welsch (1980), Regression Diagnostics, Wiley, ch. 2.
    """
    n, p, e, h, sse, _inv = _ols_influence(y, X, intercept)
    s2 = sse / (n - p)
    out = []
    for i in range(n):
        si2 = _s_deleted(sse, e[i], h[i], n, p) ** 2
        if h[i] >= 1.0 or s2 <= 0.0:
            out.append(float("nan"))
        else:
            out.append((si2 / s2) ** p / (1.0 - h[i]))
    thr = 3.0 * p / float(n)
    flagged = [1 if (v == v and abs(v - 1.0) > thr) else 0 for v in out]
    dev = [abs(v - 1.0) for v in out if v == v]
    return RichResult(payload={
        "estimate": max(dev) if dev else float("nan"),
        "covratio": out,
        "threshold": thr,
        "flagged": flagged,
        "n_influential": sum(flagged),
        "n": n,
        "p": p,
        "method": "COVRATIO deletion effect on the covariance of beta-hat",
    })


def cheatsheet():
    return "covrat: COVRATIO deletion effect on the covariance of beta-hat"
