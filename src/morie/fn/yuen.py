"""Yuen's test for two independent trimmed means.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, Sec 8.x, eq (8.12).
"""

import math

from ._richresult import RichResult
from ._rrng_core import pt, qt
from .trimse import _tmean, _winvar

__all__ = ["yuen"]


def _yuen_d(x, tr):
    n = len(x)
    h = n - 2 * math.floor(tr * n)
    return (n - 1) * _winvar(x, tr) / (h * (h - 1)), h


def yuen(x, y, tr=0.2, alpha=0.05):
    """Yuen (1974) heteroscedastic test of equal population trimmed means.

    d_j = (n_j - 1) s_wj^2 / (h_j (h_j - 1)) with h_j = n_j - 2 floor(G n_j) (8.12);
    T = (Xbar_t1 - Xbar_t2) / sqrt(d_1 + d_2) with Welch-type degrees of freedom
    (d_1 + d_2)^2 / (d_1^2/(h_1 - 1) + d_2^2/(h_2 - 1)).

    Parameters
    ----------
    x, y : sequences of float
    tr : float
    alpha : float

    Returns
    -------
    RichResult
        Keys: statistic (signed), df, diff, ci, p_value, se.

    References
    ----------
    Yuen, K. K. (1974). Biometrika 61, 165-170.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (8.12).

    Examples
    --------
    >>> r = yuen([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> r["diff"], round(r["df"], 6)
    (-2.0, 10.0)
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    d1, h1 = _yuen_d(x, tr)
    d2, h2 = _yuen_d(y, tr)
    if h1 < 2 or h2 < 2:
        raise ValueError("too few observations left after trimming")
    diff = _tmean(x, tr) - _tmean(y, tr)
    se = math.sqrt(d1 + d2)
    t = diff / se
    df = (d1 + d2) ** 2 / (d1 * d1 / (h1 - 1) + d2 * d2 / (h2 - 1))
    c = qt(1 - alpha / 2, df)
    return RichResult(
        title="Yuen's test for trimmed means",
        summary_lines=[("diff", diff), ("T", t), ("df", df)],
        payload={
            "statistic": t,
            "df": df,
            "diff": diff,
            "ci": (diff - c * se, diff + c * se),
            "p_value": 2 * pt(-abs(t), df),
            "se": se,
        },
    )


def cheatsheet():
    return "yuen: Yuen's trimmed-mean two-sample test, d_j = (n_j-1)s_wj^2/(h_j(h_j-1)). Wilcox (2017) eq (8.12)."
