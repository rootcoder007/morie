# morie.fn -- function file (rootcoder007/morie)
"""Observations tied at the combined median in the median test."""

import math

from ._richresult import RichResult

__all__ = ['medties', 'gibbons_median_ties']


def medties(x, y):
    """Median test when observations equal the combined median.

    Section 6.4 (book p. 247) defines U by a strict inequality, so any
    pooled value exactly equal to the combined median is excluded from
    the count and the effective t shrinks.  Three treatments are
    returned: ``strict`` (the book's own, exclude the ties),
    ``inclusive`` (count them as above) and ``split`` (give each
    sample its share).  When the pooled size is even and no value
    equals the median all three coincide.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``u_strict``, ``u_inclusive``, ``u_split``, ``t_strict``,
        ``t_inclusive``, ``nties``, ``median``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.4, p. 247.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    pooled = sorted(xs + ys)
    nn = m + n
    med = (
        pooled[nn // 2]
        if nn % 2
        else (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    )
    nties = sum(1 for v in pooled if v == med)
    xt = sum(1 for v in xs if v == med)
    us = sum(1 for v in xs if v > med)
    return RichResult(
        payload={
            "u_strict": float(us),
            "u_inclusive": float(us + xt),
            "u_split": float(us + xt / 2.0),
            "t_strict": int(sum(1 for v in pooled if v > med)),
            "t_inclusive": int(sum(1 for v in pooled if v >= med)),
            "nties": int(nties),
            "median": float(med),
            "m": m,
            "n": n,
            "method": "median test with ties at the combined median",
        }
    )


gibbons_median_ties = medties
