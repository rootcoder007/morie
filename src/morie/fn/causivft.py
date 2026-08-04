# morie.fn -- slice k04 (rootcoder007/morie)
"""First-stage F statistic for weak instruments.

Source FETCHED (reference implementation): the partial-F construction
used by Stock, Wright and Yogo (2002), *Journal of Business and
Economic Statistics* 20, 518-529, "A survey of weak instruments and
weak identification in generalized method of moments".  The statistic
is the ordinary F for the joint significance of the L excluded
instruments Z in the first-stage regression

    D = X_exog gamma + Z pi + v

that is, with RSS_u the residual sum of squares of that regression and
RSS_r that of the restricted fit on X_exog alone,

    F = [ (RSS_r - RSS_u) / L ] / [ RSS_u / (n - k - L) ]

on (L, n - k - L) degrees of freedom, k = ncol(X_exog) including any
intercept.  Stock-Yogo report F < 10 as the practical warning line for
a single endogenous regressor; that threshold is reported but is a rule
of thumb, not a size-correct critical value.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["causal_iv_first_stage"]


def _rss(D, y):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ beta
    return float(r @ r)


def causal_iv_first_stage(D, Z, X_exog=None, add_intercept=True):
    """First-stage F on the excluded instruments.

    Parameters
    ----------
    D : array-like, shape (n,)
        The endogenous regressor.
    Z : array-like, shape (n, L)
        Excluded instruments.
    X_exog : array-like, shape (n, q), optional
        Included exogenous covariates.
    add_intercept : bool, default True

    Returns
    -------
    RichResult
        keys: ``statistic``, ``p_value``, ``df1``, ``df2``, ``rss_u``,
        ``rss_r``, ``n``, ``n_instruments``, ``weak`` (statistic < 10),
        ``method``.
    """
    d = np.asarray(D, dtype=float).ravel()
    n = int(d.size)
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[0] != n:
        Z = Z.T
    cols = []
    if add_intercept:
        cols.append(np.ones(n))
    if X_exog is not None:
        Xe = np.atleast_2d(np.asarray(X_exog, dtype=float))
        if Xe.shape[0] != n:
            Xe = Xe.T
        cols.extend(Xe[:, j] for j in range(Xe.shape[1]))
    Dr = np.column_stack(cols) if cols else np.zeros((n, 0))
    k = int(Dr.shape[1])
    L = int(Z.shape[1])
    df2 = n - k - L
    if L < 1 or df2 < 1:
        raise ValueError("need L >= 1 and n > k + L")
    if k:
        rss_r = _rss(Dr, d)
        Du = np.column_stack([Dr, Z])
    else:
        rss_r = float(d @ d)
        Du = Z
    rss_u = _rss(Du, d)
    stat = ((rss_r - rss_u) / L) / (rss_u / df2)
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(stats.f.sf(stat, L, df2)),
            "df1": L,
            "df2": int(df2),
            "rss_u": rss_u,
            "rss_r": rss_r,
            "n": n,
            "n_instruments": L,
            "weak": bool(stat < 10.0),
            "method": "First-stage F for excluded instruments (Stock-Wright-Yogo 2002)",
        }
    )


def cheatsheet():
    return "causivft: first-stage IV F statistic for weak instruments"
