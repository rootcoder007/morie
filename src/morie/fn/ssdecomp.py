# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Decomposition of the sums of squares and the coefficient of determination.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 3.7, printed page 133 (PDF page 167), equation (3.98)::

    SQT = sum (yi - ybar)^2 = sum yi^2 - (1/n)(sum yi)^2   total scatter
    SQR = sum (yhat_i - ybar)^2                            regression
    SQE = sum (yi - yhat_i)^2 = sum eps_i^2                error
    SQT = SQR + SQE
    r^2 = SQR/SQT = 1 - SQE/SQT = B                        explained part

and the same page's (3.97), ``r^2 = b_xy b_yx = s_xy^2/(s_x^2 s_y^2) = B``
with ``0 <= B <= 1``, which is returned as ``r2_product`` so the two
routes to the same number can be compared.
"""

from __future__ import annotations

import math

from .olsfit import olsfit
from ._richresult import RichResult

__all__ = ["ssdecomp"]


def ssdecomp(x, y=None, fitted=None):
    """Split the total scatter of ``y`` into regression and error parts.

    Parameters
    ----------
    x : array-like
        Predictor, when ``fitted`` is not supplied; otherwise the
        response and ``fitted`` the fitted values.
    y : array-like, optional
        Response.  Required unless ``fitted`` is given.
    fitted : array-like, optional
        Fitted values from any model.  When supplied the decomposition
        is computed for those fitted values rather than for the simple
        regression of ``y`` on ``x``.

    Returns
    -------
    RichResult
        Keys: ``sqt``, ``sqr``, ``sqe``, ``r2``, ``r2_product``,
        ``sqt_raw``, ``n``, ``additive`` (``SQT - (SQR + SQE)``).
    """
    if fitted is not None:
        ys = [float(v) for v in x] if y is None else [float(v) for v in y]
        fh = [float(v) for v in fitted]
        r2p = None
    else:
        if y is None:
            raise ValueError("y is required unless fitted is supplied")
        fit = olsfit(x, y)
        ys = [float(v) for v in y]
        fh = fit["fitted"]
        r2p = (fit["sxy"] ** 2) / (fit["sxx"] * fit["syy"]) if fit["syy"] > 0.0 else float("nan")
    n = len(ys)
    if n != len(fh):
        raise ValueError("y and fitted must have the same length")
    if n < 2:
        raise ValueError("at least two observations are required")
    for v in ys + fh:
        if not math.isfinite(v):
            raise ValueError("y and fitted must be finite")
    ybar = sum(ys) / n
    sqt = sum((v - ybar) ** 2 for v in ys)
    sqr = sum((v - ybar) ** 2 for v in fh)
    sqe = sum((ys[i] - fh[i]) ** 2 for i in range(n))
    if sqt == 0.0:
        raise ValueError("y is constant; SQT is zero and r^2 is undefined")
    payload = {
        "sqt": sqt,
        "sqr": sqr,
        "sqe": sqe,
        "r2": sqr / sqt,
        "r2_product": r2p,
        "sqt_raw": sum(v * v for v in ys) - (sum(ys) ** 2) / n,
        "n": n,
        "additive": sqt - (sqr + sqe),
    }
    return RichResult(
        title="Sum-of-squares decomposition (Hedderich eq. 3.97/3.98)",
        summary_lines=[("SQT", sqt), ("SQR", sqr), ("SQE", sqe), ("r^2 = B", payload["r2"])],
        payload=payload,
    )


def cheatsheet() -> str:
    return "ssdecomp(x, y): SQT = SQR + SQE and r^2 = SQR/SQT -- Hedderich eq. (3.98)."
