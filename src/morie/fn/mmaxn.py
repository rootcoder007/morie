# morie.fn -- function file (rootcoder007/morie)
"""Min-max normalization for marker data or responses."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["minmax_normalization"]


def minmax_normalization(x):
    r"""Min-max normalization for marker data or responses.

    ``x* = (x - min x) / (max x - min x)`` maps the values onto ``[0, 1]``
    (Montesinos Lopez, Montesinos Lopez and Crossa 2022, ch. 2, data
    preprocessing). A 2-D input is normalised column by column; a constant
    column (``max = min``) maps to 0.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p)
        Values.

    Returns
    -------
    RichResult
        ``x_norm`` (same shape), ``min``, ``max`` (per column for 2-D).

    References
    ----------
    Montesinos Lopez, O. A., Montesinos Lopez, A. and Crossa, J. (2022). *Multivariate Statistical
    Machine Learning Methods for Genomic Prediction*. Springer, ch. 2.

    Examples
    --------
    >>> minmax_normalization([2.0, 4.0, 3.0, 10.0])["x_norm"]
    [0.0, 0.25, 0.125, 1.0]
    """
    X = x.tolist() if hasattr(x, "tolist") else list(x)
    two_d = bool(X) and isinstance(X[0], (list, tuple))
    cols = [list(map(float, c)) for c in zip(*X)] if two_d else [[float(v) for v in X]]
    out, lo, hi = [], [], []
    for c in cols:
        a, b = min(c), max(c)
        lo.append(a)
        hi.append(b)
        out.append([(v - a) / (b - a) if b > a else 0.0 for v in c])
    if two_d:
        return RichResult(payload={"x_norm": [list(r) for r in zip(*out)], "min": lo, "max": hi})
    return RichResult(payload={"x_norm": out[0], "min": lo[0], "max": hi[0]})


mmaxn = minmax_normalization


def cheatsheet():
    return "mmaxn: (x - min) / (max - min), column-wise"
