"""The error-bar overlap rule and why its standard error is wrong.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (7.36).
"""

import math

from ._richresult import RichResult

__all__ = ["ciovlap"]


def ciovlap(x, y, crit=2.0):
    """Reject equal means when the error bars Xbar_j +- crit s_j/sqrt(n_j) do not overlap.

    Rearranged, the rule rejects when |Xbar_1 - Xbar_2| / (s_1/sqrt n_1 + s_2/sqrt n_2) >= crit
    (7.36). The denominator is not the standard error of the difference,
    sqrt(s_1^2/n_1 + s_2^2/n_2), which is never larger, so the rule is conservative.

    Parameters
    ----------
    x, y : sequences of float
    crit : float
        Half-width multiplier of each error bar.

    Returns
    -------
    RichResult
        Keys: ratio (7.36), reject, welch_ratio (difference over its standard error),
        overlap (the two intervals intersect).

    References
    ----------
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (7.36).

    Examples
    --------
    >>> ciovlap([1, 2, 3], [4, 5, 6])["reject"]
    True
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("need at least 2 observations per group")

    def ms(v):
        m = sum(v) / len(v)
        return m, math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) / math.sqrt(len(v))

    m1, e1 = ms(x)
    m2, e2 = ms(y)
    ratio = abs(m1 - m2) / (e1 + e2)
    return RichResult(
        title="Error-bar overlap rule",
        summary_lines=[("ratio", ratio), ("reject", ratio >= crit)],
        payload={
            "ratio": ratio,
            "reject": ratio >= crit,
            "welch_ratio": abs(m1 - m2) / math.sqrt(e1 * e1 + e2 * e2),
            "overlap": not (m1 + crit * e1 < m2 - crit * e2 or m1 - crit * e1 > m2 + crit * e2),
        },
    )


def cheatsheet():
    return "ciovlap: error-bar overlap rule |d|/(se1 + se2) >= crit, a conservative test. Wilcox (2017) eq (7.36)."
