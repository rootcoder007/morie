"""Percentile bootstrap interval from ordered bootstrap values.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (7.15).
"""

from ._richresult import RichResult

__all__ = ["pbci"]


def pbci(values, alpha=0.05):
    """(D*_(l+1), D*_(u)) with l = alpha B / 2 rounded to the nearest integer and u = B - l (7.15).

    Parameters
    ----------
    values : sequence of float
        The B bootstrap estimates.
    alpha : float

    Returns
    -------
    RichResult
        Keys: ci, l, u, p_value (twice the smaller share of values on either side of zero).

    References
    ----------
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (7.15).

    Examples
    --------
    >>> pbci(list(range(1, 1001)))["ci"]
    (26.0, 975.0)
    """
    v = sorted(float(t) for t in values)
    B = len(v)
    if B < 2 or not 0 < alpha < 1:
        raise ValueError("need at least 2 bootstrap values and 0 < alpha < 1")
    lo = round(alpha * B / 2)
    u = B - lo
    below = sum(t < 0 for t in v) + 0.5 * sum(t == 0 for t in v)
    ph = below / B
    return RichResult(
        title="Percentile bootstrap interval",
        summary_lines=[("ci", (v[lo], v[u - 1]))],
        payload={"ci": (v[lo], v[u - 1]), "l": lo, "u": u, "p_value": 2 * min(ph, 1 - ph)},
    )


def cheatsheet():
    return "pbci: percentile bootstrap CI (D*(l+1), D*(u)), l = round(alpha B/2). Wilcox (2017) eq (7.15)."
