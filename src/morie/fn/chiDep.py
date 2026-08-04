# morie.fn -- function file (rootcoder007/morie)
"""Tail-dependence χ.

Implements sec. 8.4, p. 163-164 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["chi_dependence"]


def chi_dependence(x, y, u=0.95):
    """Empirical chi(u) = 2 - log P(F_X < u, F_Y < u) / log u
    (Coles 2001 sec. 8.4, p. 164), probabilities replaced by observed
    proportions of the rank transforms (p. 165)."""
    import math
    xs = _ev._flat(x)
    ys = _ev._flat(y)
    n = len(xs)
    if n != len(ys) or n < 4:
        raise ValueError("x and y must be equal-length, n >= 4")
    rx = _ranks01(xs)
    ry = _ranks01(ys)
    joint = sum(1 for a, b in zip(rx, ry) if a < u and b < u) / n
    joint = min(max(joint, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
    chi_u = 2.0 - math.log(joint) / math.log(u)
    chi_u = min(max(chi_u, 0.0), 1.0)      # property 1, p. 164
    res = RichResult(payload={"estimate": chi_u, "u": float(u), "n": n,
                              "method": "empirical chi(u) (Coles 2001 sec. 8.4)"})
    return with_describe_pointer(res, "chiDep")


def _ranks01(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = (k + 1.0) / (len(v) + 1.0)
    return r


def cheatsheet():
    return "chiDep: Tail-dependence χ"


# compact alias per ledger/NAMING.md
chidependence = chi_dependence
