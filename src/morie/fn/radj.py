# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Adjusted coefficient of determination for the linear model.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.2.7 "Variable Selection Method", printed page 813,
equation (8.38)::

    R2_a = 1 - [RSS / (n - (p + 1))] / [SSY / (n - 1)]
         = 1 - (n - 1)/(n - (p + 1)) (1 - R2)

with ``p`` the number of independent variables excluding the intercept,
``RSS`` the residual sum of squares and ``SSY`` the total sum of squares
about the mean.  The point the book makes in giving it: R2 never
decreases when a variable is added, so models of different size must be
compared on R2_a.

Either parameterisation may be supplied; the second printed form is the
one evaluated, with the first used to obtain R2 when RSS and SSY are
given instead.
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["radj"]


def radj(n, p, r2=None, rss=None, ssy=None):
    """Adjusted R^2 from either ``r2`` or the pair ``rss``/``ssy``.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of independent variables, excluding the intercept.
    r2 : float, optional
        Unadjusted coefficient of determination.
    rss, ssy : float, optional
        Residual and total sum of squares.  ``ssy`` must be positive.

    Returns
    -------
    RichResult
        Keys: ``radj``, ``r2``, ``n``, ``p``, ``df_resid``.
    """
    n = int(n)
    p = int(p)
    if p < 0:
        raise ValueError("p must be non-negative")
    if n - (p + 1) < 1:
        raise ValueError("n - (p + 1) must be at least 1")
    if r2 is None:
        if rss is None or ssy is None:
            raise ValueError("supply either r2, or both rss and ssy")
        rss = float(rss)
        ssy = float(ssy)
        if not math.isfinite(rss) or rss < 0.0:
            raise ValueError("rss must be finite and non-negative")
        if not math.isfinite(ssy) or ssy <= 0.0:
            raise ValueError("ssy must be finite and positive")
        r2 = 1.0 - rss / ssy
    else:
        r2 = float(r2)
        if not math.isfinite(r2):
            raise ValueError("r2 must be finite")
    val = 1.0 - (n - 1.0) / (n - (p + 1.0)) * (1.0 - r2)
    return RichResult(
        title="Adjusted coefficient of determination (Hedderich eq. 8.38)",
        summary_lines=[("Adjusted R^2", val), ("R^2", r2), ("n", n), ("p", p)],
        payload={"radj": val, "r2": r2, "n": n, "p": p, "df_resid": n - (p + 1)},
    )


def cheatsheet() -> str:
    return "radj(n, p, r2): adjusted R^2 -- Hedderich eq. (8.38)."
