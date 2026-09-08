# morie.fn -- slice s03 (rootcoder007/morie)
"""Mutual information between two discrete variables.

Source consulted: Shannon, C. E. (1948).  A mathematical theory of
communication.  *Bell System Technical Journal* 27(3), 379-423.  The
mutual information is

    I(X; Y) = sum_x sum_y p(x, y) log( p(x, y) / (p(x) p(y)) )

with the convention 0 log 0 = 0.  The 1948 paper is freely available but
was not retrievable here; the definition is quoted in its standard
published form.

The plug-in estimator is biased upward by roughly (|X| - 1)(|Y| - 1) /
(2 n) (Miller 1955), so the Miller-Madow corrected value is returned as
well -- reporting the raw plug-in alone would overstate dependence on
small samples.  Base-2 and natural units are both given.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["mutual_information"]


def _joint(a, b):
    # labels are compared as strings and sorted byte-wise, so that the R
    # mirror (sort with method = "radix") orders them identically whatever
    # the locale
    a = [str(v) for v in a]
    b = [str(v) for v in b]
    la = sorted(set(a))
    lb = sorted(set(b))
    n = len(a)
    P = [[0.0] * len(lb) for _ in range(len(la))]
    for i in range(n):
        P[la.index(a[i])][lb.index(b[i])] += 1.0 / n
    return la, lb, P


def mutual_information(y, x=None, y2=None):
    """I(X; Y) for two discrete sequences.

    Parameters
    ----------
    y : array-like
        The first variable.  (First slot, for signature stability.)
    x : array-like
        The second variable when ``y2`` is None; otherwise the first of
        the pair.
    y2 : array-like, optional
        The second of the pair.

    Returns
    -------
    estimate : I(X; Y) in nats
    bits     : the same in bits
    mm       : the Miller-Madow corrected value, in nats
    hx, hy, hxy : the marginal and joint entropies, in nats
    """
    if y2 is None:
        a = list(y)
        b = list(x)
    else:
        a = list(x)
        b = list(y2)
    n = len(a)
    la, lb, P = _joint(a, b)
    px = [0.0] * len(la)
    py = [0.0] * len(lb)
    for i in range(len(la)):
        for j in range(len(lb)):
            px[i] += P[i][j]
            py[j] += P[i][j]
    mi = 0.0
    hxy = 0.0
    for i in range(len(la)):
        for j in range(len(lb)):
            if P[i][j] > 0.0:
                mi += P[i][j] * math.log(P[i][j] / (px[i] * py[j]))
                hxy -= P[i][j] * math.log(P[i][j])
    hx = 0.0
    for v in px:
        if v > 0.0:
            hx -= v * math.log(v)
    hy = 0.0
    for v in py:
        if v > 0.0:
            hy -= v * math.log(v)
    nz = 0
    for i in range(len(la)):
        for j in range(len(lb)):
            if P[i][j] > 0.0:
                nz += 1
    mm = mi - (nz - len(la) - len(lb) + 1) / (2.0 * n) if n else float("nan")
    return RichResult(
        title="Mutual information",
        summary_lines=[("I (nats)", mi), ("I (bits)", mi / math.log(2.0))],
        payload={
            "estimate": mi,
            "mi": mi,
            "bits": mi / math.log(2.0),
            "mm": mm,
            "hx": hx,
            "hy": hy,
            "hxy": hxy,
            "n": n,
            "method": "Plug-in mutual information (Shannon 1948) with the Miller-Madow correction",
        },
    )


def cheatsheet():
    return "mutifo: Mutual information between X and Y"
