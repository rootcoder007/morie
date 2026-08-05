# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Goodness of fit of a loglinear model: Pearson chi-squared and G^2.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.5 "Poisson Regression and Loglinear Models", printed page 851,
equations (8.95) and (8.96)::

    chi2_hat = sum_{i,j} (y_ij - n pihat_ij)^2 / (n pihat_ij)         (8.95)

    G^2 = 2 sum_{i,j} y_ij log( y_ij / (n pihat_ij) )                 (8.96)

with ``sum y_ij = n`` and ``pihat_ij`` the cell probabilities estimated
under the model.  The book calls (8.95) the Pearson-residual statistic
and (8.96) the likelihood-quotient statistic, notes that G^2 is usually
preferred because its minimum comes from the maximum-likelihood
estimate, and states that both are asymptotically chi-squared, so a
model can be evaluated by a p-value.

The default model is independence (8.97), ``log n pi_ij = mu + lam_i^X
+ lam_j^Y``, whose fitted cell counts are the familiar row-times-column
product over the grand total and whose degrees of freedom are
``(k1 - 1)(k2 - 1)``.  Fitted counts for any other model may be passed
directly in ``expected`` with the matching ``df``.

``0 log 0`` is taken as 0, its limiting value.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _tail1core as C
from ._richresult import hypothesis_test_result

__all__ = ["llgsq"]


def _asmatrix(x):
    rows = [[float(v) for v in row] for row in x]
    if not rows:
        raise ValueError("the table must not be empty")
    k = len(rows[0])
    if k == 0:
        raise ValueError("the table must not be empty")
    for r in rows:
        if len(r) != k:
            raise ValueError("every row of the table must have the same length")
    return rows


def llgsq(observed, expected=None, df=None, alpha=0.05):
    """Loglinear goodness of fit, defaulting to the independence model.

    Parameters
    ----------
    observed : 2-D array-like
        Contingency table of non-negative counts.
    expected : 2-D array-like, optional
        Fitted cell counts ``n pihat_ij`` under the model under test.
        Defaults to the independence fit.
    df : int, optional
        Degrees of freedom.  Required when ``expected`` is supplied;
        defaults to ``(k1 - 1)(k2 - 1)`` for the independence model.
    alpha : float
        Significance level for the reject decision.

    Returns
    -------
    RichResult
        statistic = G^2.  Keys: ``g2``, ``chisq``, ``expected``, ``df``,
        ``pvalue``, ``pvalue_chisq``, ``reject``, ``n``.
    """
    o = _asmatrix(observed)
    k1 = len(o)
    k2 = len(o[0])
    total = 0.0
    for r in o:
        for v in r:
            if not math.isfinite(v) or v < 0.0:
                raise ValueError("every observed count must be finite and non-negative")
            total += v
    if total <= 0.0:
        raise ValueError("the table must contain at least one positive count")
    if expected is None:
        if k1 < 2 or k2 < 2:
            raise ValueError("the independence model needs a table of at least 2 by 2")
        rows = [sum(r) for r in o]
        cols = [sum(o[i][j] for i in range(k1)) for j in range(k2)]
        e = [[rows[i] * cols[j] / total for j in range(k2)] for i in range(k1)]
        if df is None:
            df = (k1 - 1) * (k2 - 1)
    else:
        e = _asmatrix(expected)
        if len(e) != k1 or len(e[0]) != k2:
            raise ValueError("observed and expected must have the same shape")
        if df is None:
            raise ValueError("df is required when expected is supplied")
    df = int(df)
    if df < 1:
        raise ValueError("df must be at least 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    chi = 0.0
    g2 = 0.0
    for i in range(k1):
        for j in range(k2):
            if not math.isfinite(e[i][j]) or e[i][j] <= 0.0:
                raise ValueError("every expected count must be finite and positive")
            chi += (o[i][j] - e[i][j]) ** 2 / e[i][j]
            if o[i][j] > 0.0:
                g2 += 2.0 * o[i][j] * math.log(o[i][j] / e[i][j])
    pg = 1.0 - C.pchisq(g2, df)
    pc = 1.0 - C.pchisq(chi, df)
    return hypothesis_test_result(
        test_name="Loglinear model goodness of fit (Hedderich eqs. 8.95, 8.96)",
        statistic=g2,
        pvalue=pg,
        df=df,
        alpha=float(alpha),
        extra_summary=[("Pearson chi-squared", chi), ("p (chi-squared)", pc)],
        extra_payload={
            "g2": g2,
            "chisq": chi,
            "expected": e,
            "pvalue_chisq": pc,
            "reject": bool(pg < float(alpha)),
            "n": total,
        },
        callable_name="llgsq",
    )


def cheatsheet() -> str:
    return "llgsq(observed): loglinear chi-squared and G^2 -- Hedderich eqs. (8.95), (8.96)."
