"""Standard error of the trimmed mean via the Winsorized variance.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eqs (4.9)-(4.10).
"""

import math

from ._richresult import RichResult

__all__ = ["trimse"]


def _tmean(x, tr):
    s = sorted(float(v) for v in x)
    g = math.floor(tr * len(s))
    return sum(s[g : len(s) - g]) / (len(s) - 2 * g)


def _winvar(x, tr):
    """Winsorized sample variance (n - 1 divisor), g = floor(tr n) values pulled in on each side."""
    s = sorted(float(v) for v in x)
    n = len(s)
    g = math.floor(tr * n)
    w = [min(max(v, s[g]), s[n - g - 1]) for v in s]
    m = sum(w) / n
    return sum((v - m) ** 2 for v in w) / (n - 1)


def trimse(x, tr=0.2):
    """Estimated standard error of the tr-trimmed mean: s_w / ((1 - 2 tr) sqrt(n)).

    With 20% trimming this is s_w^2 / (0.6^2 n) squared (4.9); in general the
    0.6 becomes 1 - 2G (4.10), where s_w^2 is the Winsorized sample variance.

    Parameters
    ----------
    x : sequence of float
    tr : float
        Trimming proportion in each tail, 0 <= tr < 0.5.

    Returns
    -------
    RichResult
        Keys: se, trimmed_mean, winsorized_variance, n.

    References
    ----------
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eqs (4.9)-(4.10).

    Examples
    --------
    >>> round(trimse([2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.2, 2.2, 5.0, 40])["se"], 9)
    0.607161043
    """
    x = [float(v) for v in x]
    n = len(x)
    if n < 2 or not 0 <= tr < 0.5:
        raise ValueError("need n >= 2 and 0 <= tr < 0.5")
    wv = _winvar(x, tr)
    se = math.sqrt(wv) / ((1 - 2 * tr) * math.sqrt(n))
    return RichResult(
        title="Standard error of the trimmed mean",
        summary_lines=[("se", se), ("trimmed mean", _tmean(x, tr))],
        payload={"se": se, "trimmed_mean": _tmean(x, tr), "winsorized_variance": wv, "n": n},
    )


def cheatsheet():
    return "trimse: s_w / ((1 - 2 tr) sqrt n), the trimmed-mean standard error. Wilcox (2017) eqs (4.9)-(4.10)."
