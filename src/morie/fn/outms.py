# morie.fn -- wave 4 slice b_3 (rootcoder007/morie)
"""Outlier detection based on the sample mean and standard deviation.

Source: Wilcox, R. R. (2017), *Modern Statistics for the Social and
Behavioral Sciences: A Practical Introduction*, 2nd edn, CRC Press,
section 2.5.1, equation (2.13), p.32.  Declare X an outlier if

    |X - Xbar| / s > 2.                                        (2.13)

This rule is included because it is commonly used, not because it is
good: both Xbar and s have a breakdown point of 1/n, so the outliers
inflate the very yardstick that is meant to find them.  Wilcox uses it
to demonstrate *masking* -- see the second anchor below, where adding a
second, larger outlier stops the first from being detected.

Anchors (printed worked examples, p.32):
  * 2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,1000 has Xbar = 65.94, s = 249.1 and
    |1000 - 65.94| / 249.1 = 3.75, so 1000 is declared an outlier;
  * appending 10000 gives Xbar = 650.3, s = 2421.4 and
    |1000 - Xbar| / s = 0.14, so 1000 is *not* declared an outlier.
"""

from __future__ import annotations

from math import sqrt

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["outms"]

_METHOD = "Wilcox (2017) mean/SD outlier rule, eq. (2.13)"


def outms(x, crit=2.0):
    """Flag outliers with the mean-and-standard-deviation rule (2.13).

    Parameters
    ----------
    x : array-like
        The sample; at least two observations.
    crit : float, default 2.0
        The cut-off of equation (2.13).  Must be positive.

    Returns
    -------
    result : RichResult
        Keys: flag (0/1 per observation), which (1-based positions of
        the outliers), out_val, n_out, center, scale, dis, crit, n,
        estimate (= n_out), method.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("outms: need at least 2 observations")
    for v in xs:
        if v != v:
            raise ValueError("outms: x contains a missing value")
    crit = float(crit)
    if not (crit > 0.0):
        raise ValueError("outms: crit must be positive")
    center = 0.0
    for v in xs:
        center += v
    center /= n
    ss = 0.0
    for v in xs:
        ss += (v - center) ** 2
    scale = sqrt(ss / (n - 1))
    if not (scale > 0.0):
        raise ValueError("outms: the standard deviation is zero")
    dis = [abs(v - center) / scale for v in xs]
    flag = [1 if d > crit else 0 for d in dis]
    which = [i + 1 for i in range(n) if flag[i] == 1]
    out_val = [xs[i - 1] for i in which]
    return RichResult(
        title="Mean/SD outlier rule",
        summary_lines=[("n", n), ("n_out", len(which))],
        payload={
            "flag": flag,
            "which": which,
            "out_val": out_val,
            "n_out": len(which),
            "center": center,
            "scale": scale,
            "dis": dis,
            "crit": crit,
            "n": n,
            "estimate": float(len(which)),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "outms: Wilcox mean/SD outlier rule (eq. 2.13)"
