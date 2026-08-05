# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Simple linear regression by the normal equations.

Source READ FROM THE CORPUS PDF, pages rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 3.7.9, printed page 132 (PDF page 166), equation (3.96)::

        n sum(xi yi) - sum(xi) sum(yi)     s_xy
    b = ------------------------------  = ------                   (3.96)
          n sum(xi^2) - (sum(xi))^2        s_x^2

    a = ybar - b xbar

and again, derived from the partial derivatives of the error sum of
squares, section 6.4, printed page 347 (PDF page 381), equation (6.25)::

    dS/dalpha = -2 sum(yi - alpha - beta xi)     = 0
    dS/dbeta  = -2 sum(yi - alpha - beta xi) xi  = 0

    betahat  = sum((xi - xbar)(yi - ybar)) / sum((xi - xbar)^2)
             = s_xy / (s_x)^2
    alphahat = ybar - betahat xbar                                  (6.25)

(3.96) and (6.25) are the same estimator written two ways -- the raw
cross-product form and the centred form -- so they are one method.  The
centred form is used here because it is the numerically stabler of the
two; ``sxy_raw`` and ``sxx_raw`` are also returned so the raw form of
(3.96) can be read off directly.

Book worked example, printed page 132/133: lung-tumour formation against
asbestos exposure,
``x = 50, 400, 500, 900, 1100, 1600, 1800, 2000, 3000``,
``y = 2, 6, 5, 10, 26, 42, 37, 28, 50``, for which the book's own
``lm(lungca ~ asbestos)`` prints intercept ``0.54047`` and slope
``0.01772``.
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["olsfit"]


def olsfit(x, y):
    """Least-squares intercept and slope of ``y`` on ``x``.

    Parameters
    ----------
    x, y : array-like
        Paired observations, same length, at least two points and ``x``
        not constant.

    Returns
    -------
    RichResult
        Keys: ``intercept``, ``slope``, ``n``, ``xbar``, ``ybar``,
        ``sxy``, ``sxx``, ``syy``, ``sxy_raw``, ``sxx_raw``, ``cov``,
        ``var_x``, ``fitted``, ``residuals``.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    n = len(xs)
    if n != len(ys):
        raise ValueError("x and y must have the same length")
    if n < 2:
        raise ValueError("at least two observations are required")
    for v in xs + ys:
        if not math.isfinite(v):
            raise ValueError("x and y must be finite")
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((v - xbar) ** 2 for v in xs)
    sxy = sum((xs[i] - xbar) * (ys[i] - ybar) for i in range(n))
    syy = sum((v - ybar) ** 2 for v in ys)
    if sxx == 0.0:
        raise ValueError("x is constant; the slope is not identified")
    b = sxy / sxx
    a = ybar - b * xbar
    fitted = [a + b * v for v in xs]
    payload = {
        "intercept": a,
        "slope": b,
        "n": n,
        "xbar": xbar,
        "ybar": ybar,
        "sxy": sxy,
        "sxx": sxx,
        "syy": syy,
        "sxy_raw": n * sum(xs[i] * ys[i] for i in range(n)) - sum(xs) * sum(ys),
        "sxx_raw": n * sum(v * v for v in xs) - sum(xs) ** 2,
        "cov": sxy / (n - 1),
        "var_x": sxx / (n - 1),
        "fitted": fitted,
        "residuals": [ys[i] - fitted[i] for i in range(n)],
    }
    return RichResult(
        title="Least-squares straight line (Hedderich eq. 3.96 / 6.25)",
        summary_lines=[("intercept a", a), ("slope b", b), ("n", n)],
        payload=payload,
    )


def cheatsheet() -> str:
    return "olsfit(x, y): intercept and slope by the normal equations -- Hedderich eq. (3.96)/(6.25)."
