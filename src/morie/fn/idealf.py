# morie.fn -- wave 4 slice b_3 (rootcoder007/morie)
"""Ideal fourths: the lower and upper quartile estimates q1, q2 and the IQR.

Source: Wilcox, R. R. (2017), *Modern Statistics for the Social and
Behavioral Sciences: A Practical Introduction*, 2nd edn, CRC Press,
section 2.4.3, equations (2.6), (2.7) and (2.8), p.27.

With the observations written in ascending order X_(1) <= ... <= X_(n),
let j be the integer portion of n/4 + 5/12 and h = n/4 + 5/12 - j.  The
lower ideal fourth is

    q1 = (1 - h) X_(j) + h X_(j+1)                             (2.6)

and, with k = n - j + 1, the upper ideal fourth is

    q2 = (1 - h) X_(k) + h X_(k-1)                             (2.7)

in which case the interquartile range is

    IQR = q2 - q1.                                             (2.8)

Note the *descending* indexing in (2.7): the upper fourth interpolates
from X_(k) back towards X_(k-1), so that q2 is the mirror image of q1.
Writing X_(k+1) there instead -- the obvious slip -- breaks that
symmetry and is caught by the anchor below.

Anchor (printed worked example, p.27): for the twelve values -29.6,
-20.9, -19.7, -15.4, -12.3, -8.0, -4.3, 0.8, 2.0, 6.2, 11.2, 25.0 the
book gets n/4 + 5/12 = 3.41667, j = 3, h = 0.41667 and q1 = -17.9.
"""

from __future__ import annotations

from math import floor

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["idealf"]

_METHOD = "Wilcox (2017) ideal fourths, eq. (2.6)-(2.8)"


def _fourths(x):
    """Return (q1, q2, j, h) for the already-validated sample ``x``."""
    xs = sorted(float(v) for v in x)
    n = len(xs)
    g = n / 4.0 + 5.0 / 12.0
    j = int(floor(g))
    h = g - j
    kk = n - j + 1
    q1 = (1.0 - h) * xs[j - 1] + h * xs[j]
    q2 = (1.0 - h) * xs[kk - 1] + h * xs[kk - 2]
    return q1, q2, j, h


def idealf(x):
    """Lower and upper ideal fourths and the interquartile range.

    Parameters
    ----------
    x : array-like
        The sample.  At least three observations are needed, otherwise
        the index j of equation (2.6) falls below one.

    Returns
    -------
    result : RichResult
        Keys: q1, q2, iqr, j, h, n, estimate (= q1), method.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 3:
        raise ValueError("idealf: need at least 3 observations")
    for v in xs:
        if v != v:
            raise ValueError("idealf: x contains a missing value")
    q1, q2, j, h = _fourths(xs)
    return RichResult(
        title="Ideal fourths",
        summary_lines=[("n", n), ("q1", q1), ("q2", q2)],
        payload={
            "q1": q1,
            "q2": q2,
            "iqr": q2 - q1,
            "j": j,
            "h": h,
            "n": n,
            "estimate": q1,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "idealf: Wilcox ideal fourths q1, q2 and the interquartile range"
