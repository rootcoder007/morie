"""Explanatory power and strength of association.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (14.10).
"""

import math

from ._richresult import RichResult
from .trimse import _winvar

__all__ = ["explpow"]


def explpow(y, yhat, measure="variance", tr=0.2):
    """eta^2 = tau^2(Yhat) / tau^2(Y) for a measure of variation tau^2 (14.10).

    With the ordinary variance and least-squares fitted values eta^2 is R^2;
    ``measure="winsorized"`` uses the tr-Winsorized variance. The strength of
    association is eta = sqrt(eta^2).

    Parameters
    ----------
    y, yhat : sequences of float
        Outcomes and fitted values from any regression estimator.
    measure : {"variance", "winsorized"}
    tr : float
        Winsorizing proportion when measure="winsorized".

    Returns
    -------
    RichResult
        Keys: eta2, eta.

    References
    ----------
    Doksum, K. A. & Samarov, A. (1995). Annals of Statistics 23, 1443-1473.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (14.10).

    Examples
    --------
    >>> explpow([1, 2, 3, 4], [1, 2, 3, 4])["eta2"]
    1.0
    """
    y = [float(v) for v in y]
    f = [float(v) for v in yhat]
    if len(y) != len(f) or len(y) < 2:
        raise ValueError("y and yhat must have equal length >= 2")
    if measure == "variance":

        def tau2(v):
            m = sum(v) / len(v)
            return sum((a - m) ** 2 for a in v) / (len(v) - 1)
    elif measure == "winsorized":

        def tau2(v):
            return _winvar(v, tr)
    else:
        raise ValueError('measure must be "variance" or "winsorized"')
    e2 = tau2(f) / tau2(y)
    return RichResult(
        title="Explanatory power", summary_lines=[("eta^2", e2)], payload={"eta2": e2, "eta": math.sqrt(e2)}
    )


def cheatsheet():
    return "explpow: explanatory power tau^2(Yhat)/tau^2(Y). Wilcox (2017) eq (14.10)."
