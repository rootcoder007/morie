"""Quantile of the studentized range (port of R's nmath qtukey).

Used for Tukey-Kramer intervals, Wilcox (2017), Modern Statistics for the Social and
Behavioral Sciences, eq (12.4).
"""

import math

from ._richresult import RichResult
from .ptukey import _ptukey

__all__ = ["qtukey"]


def _qinv(p, c, v):
    ps = 0.5 - 0.5 * p
    yi = math.sqrt(math.log(1.0 / (ps * ps)))
    t = yi + (
        (((yi * -0.453642210148e-04 + -0.204231210125) * yi + -0.342242088547) * yi + -1.0) * yi + 0.322232421088
    ) / (
        (((yi * 0.38560700634e-02 + 0.103537752850) * yi + 0.531103462366) * yi + 0.588581570495) * yi
        + 0.993484626060e-01
    )
    if v < 120.0:
        t += (t * t * t + t) / v / 4.0
    q = 0.8832 - 0.2368 * t
    if v < 120.0:
        q += -1.214 / v + 1.208 * t / v
    return t * (q * math.log(c - 1.0) + 1.4142)


def qtukey(p, nmeans, df, nranges=1):
    """Studentized range quantile by R's secant iteration (stops when iterates differ by < 1e-4).

    Parameters
    ----------
    p : float
        Lower-tail probability, 0 < p < 1.
    nmeans : int
    df : float
    nranges : int

    Returns
    -------
    RichResult
        Keys: q.

    References
    ----------
    Copenhaver, M. D. & Holland, B. S. (1988). Journal of Statistical
    Computation and Simulation 30, 1-15.

    Examples
    --------
    >>> round(qtukey(0.95, 3, 20)["q"], 4)
    3.5779
    """
    if not 0 < p < 1 or df < 2 or nmeans < 2 or nranges < 1:
        raise ValueError("need 0 < p < 1, df >= 2, nmeans >= 2 and nranges >= 1")
    rr, cc, df = float(nranges), float(nmeans), float(df)
    x0 = _qinv(p, cc, df)
    valx0 = _ptukey(x0, rr, cc, df) - p
    x1 = max(0.0, x0 - 1.0) if valx0 > 0.0 else x0 + 1.0
    valx1 = _ptukey(x1, rr, cc, df) - p
    ans = x1
    for _ in range(1, 50):
        ans = x1 - valx1 * (x1 - x0) / (valx1 - valx0)
        valx0 = valx1
        x0 = x1
        if ans < 0.0:
            ans = 0.0
        valx1 = _ptukey(ans, rr, cc, df) - p
        x1 = ans
        if abs(x1 - x0) < 0.0001:
            break
    return RichResult(title="Studentized range quantile", summary_lines=[("q", ans)], payload={"q": ans})


def cheatsheet():
    return "qtukey: studentized range quantile, port of R's qtukey."
