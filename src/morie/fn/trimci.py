"""Tukey-McLaughlin test and confidence interval for a trimmed mean.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (4.11).
"""

import math

from ._richresult import RichResult
from ._rrng_core import pt, qt
from .trimse import _tmean, _winvar

__all__ = ["trimci"]


def trimci(x, tr=0.2, alpha=0.05, null_value=0.0):
    """Tukey-McLaughlin inference for the population trimmed mean.

    T = (Xbar_t - null) / (s_w / ((1 - 2G) sqrt n)) (4.11) is referred to
    Student's t with n - 2g - 1 degrees of freedom, g = floor(G n); the
    1 - alpha interval is Xbar_t +- t_{1-alpha/2} s_w / ((1 - 2G) sqrt n).

    Parameters
    ----------
    x : sequence of float
    tr : float
        Trimming proportion G.
    alpha : float
    null_value : float

    Returns
    -------
    RichResult
        Keys: estimate, se, statistic, df, ci, p_value.

    References
    ----------
    Tukey, J. W. & McLaughlin, D. H. (1963). Sankhya A 25, 331-352.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (4.11).

    Examples
    --------
    >>> r = trimci([2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.2, 2.2, 5.0, 40], null_value=3)
    >>> round(r["statistic"], 6), r["df"]
    (1.427408, 8)
    """
    x = [float(v) for v in x]
    n = len(x)
    g = math.floor(tr * n)
    df = n - 2 * g - 1
    if df < 1 or not 0 <= tr < 0.5:
        raise ValueError("too few observations left after trimming")
    est = _tmean(x, tr)
    se = math.sqrt(_winvar(x, tr)) / ((1 - 2 * tr) * math.sqrt(n))
    t = (est - null_value) / se
    c = qt(1 - alpha / 2, df)
    p = 2 * pt(-abs(t), df)
    return RichResult(
        title="Tukey-McLaughlin trimmed-mean inference",
        summary_lines=[("trimmed mean", est), ("T", t), ("df", df), ("p", p)],
        payload={"estimate": est, "se": se, "statistic": t, "df": df, "ci": (est - c * se, est + c * se), "p_value": p},
    )


def cheatsheet():
    return "trimci: Tukey-McLaughlin T and CI for a trimmed mean, df = n - 2g - 1. Wilcox (2017) eq (4.11)."
