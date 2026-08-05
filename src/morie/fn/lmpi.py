# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Prediction and mean-response intervals for the linear regression model.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.2.6, printed page 813, equations (8.36) and (8.37)::

    yhat0 +/- t_{n-p-1, 1-alpha/2} sigmahat sqrt(1 + x0' (X'X)^-1 x0)  (8.36)

    yhat0 +/- t_{n-p-1, 1-alpha/2} sigmahat sqrt(    x0' (X'X)^-1 x0)  (8.37)

(8.36) is the interval for a single future observation; (8.37) drops the
leading 1 and is the narrower interval for a mean future value, because
it carries no Var(epsilon) term.  ``sigmahat`` is the residual standard
deviation ``sqrt(RSS / (n - p - 1))`` and ``p`` is the number of
predictors excluding the intercept.

The book worked example on the same page fits litter size on body weight
and brain weight (n = 20, p = 2) and reports for
``x0 = (1, 8.0, 0.4)`` the estimate ``yhat0 = 6.37`` with 95% prediction
interval ``4.49 ... 8.24``, rounded outward to 4.4 ... 8.3.

The design matrix is passed exactly as the book builds it, with its own
leading column of ones, so that ``x0`` is written the same way.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["lmpi"]


def _qt(p, df):
    """Quantile of Student t by bisection on the CDF in ``_tail1core``.

    Base R has ``qt``; the Python side is de-externalised, so the
    quantile is obtained by inverting the same ``pt`` both arms agree
    on.  200 bisections on a bracket of half-width 1e4 leave a residual
    below 1e-56, far under the 1e-9 parity tolerance.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")
    lo, hi = -1.0e4, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if C.pt(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lmpi(X, y, x0, level=0.95, mean=False):
    """Prediction interval (8.36) or mean-response interval (8.37).

    Parameters
    ----------
    X : 2-D array-like
        Design matrix INCLUDING its own leading column of ones, as the
        book builds it with ``cbind(1, ...)``.
    y : array-like
        Response, length ``nrow(X)``.
    x0 : array-like
        The new design row, same width as ``X`` and likewise starting
        with 1.
    level : float
        Confidence level.
    mean : bool
        ``False`` (default) gives (8.36), a single future observation;
        ``True`` gives (8.37), a mean future value.

    Returns
    -------
    RichResult
        Keys: ``fit``, ``lower``, ``upper``, ``se_fit``, ``sigma``,
        ``df``, ``tquant``, ``leverage``, ``coef``, ``level``, ``mean``.
    """
    rows = [[float(v) for v in r] for r in X]
    if not rows:
        raise ValueError("X must not be empty")
    n = len(rows)
    k = len(rows[0])
    for r in rows:
        if len(r) != k:
            raise ValueError("every row of X must have the same length")
    yv = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    if len(yv) != n:
        raise ValueError("X and y must have the same number of rows")
    xn = [float(v) for v in np.asarray(x0, dtype=float).ravel()]
    if len(xn) != k:
        raise ValueError("x0 must have the same length as a row of X")
    df = n - k
    if df < 1:
        raise ValueError("need more observations than columns of X")
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be strictly between 0 and 1")
    beta = C.lstsq(rows, yv)
    beta = [float(v) for v in beta]
    rss = 0.0
    for i in range(n):
        fit_i = 0.0
        for j in range(k):
            fit_i += rows[i][j] * beta[j]
        rss += (yv[i] - fit_i) ** 2
    sigma = math.sqrt(rss / df)
    xtx = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xtxinv = C.inv(xtx)
    lev = 0.0
    for a in range(k):
        for b in range(k):
            lev += xn[a] * float(xtxinv[a][b]) * xn[b]
    if lev < 0.0:
        lev = 0.0
    fit = sum(xn[j] * beta[j] for j in range(k))
    root = math.sqrt(lev if mean else 1.0 + lev)
    tq = _qt(0.5 + float(level) / 2.0, df)
    half = tq * sigma * root
    return RichResult(
        title="Linear model prediction interval (Hedderich eqs. 8.36, 8.37)",
        summary_lines=[("Fit", fit), ("Lower", fit - half), ("Upper", fit + half)],
        payload={
            "fit": fit,
            "lower": fit - half,
            "upper": fit + half,
            "se_fit": sigma * root,
            "sigma": sigma,
            "df": df,
            "tquant": tq,
            "leverage": lev,
            "coef": beta,
            "level": float(level),
            "mean": bool(mean),
        },
    )


def cheatsheet() -> str:
    return "lmpi(X, y, x0): linear-model prediction interval -- Hedderich eqs. (8.36), (8.37)."
