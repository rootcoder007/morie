# morie.fn -- function file (rootcoder007/morie)
"""Negative control outcome: estimate the known-null effect and test it."""

import math

from . import _tail1core as C
from .negct import negative_control

from ._richresult import RichResult

__all__ = ["negative_control_outcome", "negativecontroloutcome"]


def negative_control_outcome(y_neg, D, X=None, alpha=0.05):
    """Fit the effect of ``D`` on a known-null outcome and test it for zero.

    A negative control outcome is one the exposure cannot plausibly
    affect but that shares the exposure's confounders.  Its true effect
    is zero by construction, so any effect the model finds there is
    bias, not signal.  The logic is one-directional and worth stating
    plainly: a non-zero estimate is evidence of residual confounding or
    misspecification, while a zero estimate is NOT evidence of
    unconfoundedness -- it only says this particular control did not
    detect a problem.

    The estimate is the OLS coefficient on ``D`` in

        y_neg = b0 + b_D D + X gamma + e

    with the usual homoscedastic standard error
    ``sqrt(s^2 (X'X)^-1_DD)``, ``s^2 = RSS / (n - p)``.  The DECISION is
    not recomputed here: the z-test and its "confounding suspected"
    verdict are delegated to ``negct.negative_control``, which is the
    single implementation of that test in this package.

    Parameters
    ----------
    y_neg : array-like, length n
        Negative-control outcome.
    D : array-like, length n
        Exposure.
    X : array-like, shape (n, q), optional
        Adjustment covariates.
    alpha : float, default 0.05
        Size of the test.  Note that ``negct`` fixes its own verdict at
        the 5% level; ``confounding_suspected`` here uses ``alpha``, and
        both are returned so the difference is visible.

    Returns
    -------
    RichResult
        ``estimate`` (b_D), ``se``, ``z``, ``p_value``,
        ``confounding_suspected``, ``ci_lower``, ``ci_upper``, ``n``,
        ``p``.

    References
    ----------
    Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010), "Negative
    controls: a tool for detecting confounding and bias in observational
    studies", Epidemiology 21(3), 383-388,
    doi:10.1097/EDE.0b013e3181d61eeb, verified against Crossref.
    Shi, X., Miao, W. and Tchetgen Tchetgen, E. (2020), "A selective
    review of negative control methods in epidemiology", Current
    Epidemiology Reports 7, 190-202, doi:10.1007/s40471-020-00243-4.
    Neither paper was in the local corpus; the estimator above is
    ordinary least squares and the test is the standard Wald one.
    """
    y = C.vec(y_neg)
    d = C.vec(D)
    n = len(y)
    if n == 0:
        raise ValueError("y_neg is empty")
    if len(d) != n:
        raise ValueError("y_neg and D must have the same length")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if X is None:
        cols = []
    else:
        Xm = C.mat(X)
        if len(Xm) != n:
            raise ValueError("X must have one row per observation")
        cols = [[row[j] for row in Xm] for j in range(len(Xm[0]))]
    dm = [[1.0, d[i]] + [c[i] for c in cols] for i in range(n)]
    p = len(dm[0])
    if n <= p:
        raise ValueError("need more observations than parameters")
    beta, fitted, resid, xtxinv = C.lstsq(dm, y)
    rss = sum(t * t for t in resid)
    s2 = rss / (n - p)
    var = s2 * xtxinv[1][1]
    se = math.sqrt(var) if var > 0.0 else 0.0
    b = float(beta[1])
    if se <= 0.0:
        raise ValueError("the exposure is collinear with the adjustment set; "
                         "its coefficient has no standard error")
    # The verdict is negct's, not a second implementation of it.
    t = negative_control(b, b, se_estimate=se, se_negative=se)
    z = float(t.statistic)
    pv = float(t.p_value)
    zc = _qnorm_upper(a / 2.0)
    return RichResult(payload={
        "estimate": b, "se": se, "z": z, "p_value": pv,
        "confounding_suspected": 1.0 if pv < a else 0.0,
        "negct_verdict_at_5pct": 1.0 if t.extra["confounding_suspected"] else 0.0,
        "ci_lower": b - zc * se, "ci_upper": b + zc * se,
        "alpha": a, "n": n, "p": p,
        "method": "Negative control outcome (Wald test of a known null)"})


def _qnorm_upper(q):
    """Upper-tail standard normal quantile, via the shared pnorm."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - C.pnorm(mid) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


negativecontroloutcome = negative_control_outcome


def cheatsheet():
    return "negctc: negative control outcome, OLS effect plus negct's verdict"
