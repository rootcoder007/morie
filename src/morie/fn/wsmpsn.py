# morie.fn -- function file (rootcoder007/morie)
"""Sample correlation with its test and interval."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pearsonr", "wasserman_pearson_corr"]


def pearsonr(x, y, level=0.95):
    """Sample correlation, its t test, and a Fisher-z interval.

    Two different approximations are in play and it is worth keeping
    them apart: the t test is exact under bivariate normality with
    df = n - 2, while the interval comes from Fisher's variance-
    stabilising transform, which is why the interval is not symmetric
    about r.  Neither is a test of independence outside the normal
    model -- correlation zero and independence are different claims.

    Formula: rhat = sum (x_i - xbar)(y_i - ybar)
                    / sqrt( sum (x_i - xbar)^2 sum (y_i - ybar)^2 );
             t = rhat sqrt(n - 2)/sqrt(1 - rhat^2) on n - 2 df;
             z = atanh(rhat), se_z = 1/sqrt(n - 3)

    Parameters
    ----------
    x, y : array-like
        Paired samples of the same length, n >= 3.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``statistic``, ``p_value``, ``df``, ``z``,
        ``se_z``, ``ci_lower``, ``ci_upper``, ``n``.

    References
    ----------
    Wasserman (2004), All of Statistics, Example 7.13, which derives
    the plug-in estimate rhat = sum (X_i - Xbar)(Y_i - Ybar) /
    sqrt(sum (X_i - Xbar)^2 sum (Y_i - Ybar)^2) and calls it the
    sample correlation.  Fetched as the full text of the book.  The
    t test on n - 2 degrees of freedom and Fisher's z transform are
    NOT in that section; they are the standard published forms (Fisher,
    1915, Biometrika 10(4), 507-521).
    """
    x = C.vec(x)
    y = C.vec(y)
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    if n < 3:
        raise ValueError("n must be at least 3 for the z interval")
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        raise ValueError("a sample with zero variance has no correlation")
    r = sxy / math.sqrt(sxx * syy)
    r = min(1.0, max(-1.0, r))
    df = n - 2
    if abs(r) >= 1.0:
        t = math.inf if r > 0 else -math.inf
        p = 0.0
    else:
        t = r * math.sqrt(df) / math.sqrt(1.0 - r * r)
        p = 2.0 * (1.0 - C.pt(abs(t), df))
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 1.0 else math.copysign(math.inf, r)
    sez = 1.0 / math.sqrt(n - 3)
    zc = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": r, "statistic": t, "p_value": p, "df": float(df),
        "z": z, "se_z": sez,
        "ci_lower": math.tanh(z - zc * sez), "ci_upper": math.tanh(z + zc * sez),
        "n": n, "method": "Sample correlation, Wasserman Example 7.13"})


wasserman_pearson_corr = pearsonr


def cheatsheet():
    return "wsmpsn: rhat = Sxy/sqrt(Sxx Syy); t on n-2 df; Fisher-z CI"
