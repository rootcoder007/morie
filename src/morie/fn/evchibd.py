# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic-independence diagnostic χ̄(u).

Implements sec. 8.4, p. 164 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_chibar_dependence"]


def evt_chibar_dependence(x, y, u_grid=None):
    """chibar(u) = 2 log(1-u) / log P(F_X > u, F_Y > u) - 1
    (Coles 2001 sec. 8.4, p. 164). chibar -> 1 signals asymptotic
    dependence; |chibar| < 1 asymptotic independence (p. 165)."""
    import math
    xs = _ev._flat(x)
    ys = _ev._flat(y)
    n = len(xs)
    if n != len(ys) or n < 4:
        raise ValueError("x and y must be equal-length, n >= 4")
    from .chiDep import _ranks01
    rx = _ranks01(xs)
    ry = _ranks01(ys)
    if u_grid is None:
        u_grid = [0.5 + 0.45 * k / 19.0 for k in range(20)]
    curve = []
    for u in u_grid:
        joint = sum(1 for a, b in zip(rx, ry)
                    if a > u and b > u) / n
        joint = min(max(joint, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
        cb = 2.0 * math.log(1.0 - u) / math.log(joint) - 1.0
        curve.append(min(max(cb, -1.0), 1.0))   # property 1, p. 164
    res = RichResult(payload={"chibar_curve": curve,
                              "u_grid": [float(u) for u in u_grid],
                              "n": n,
                              "method": "empirical chibar(u) (Coles 2001 sec. 8.4)"})
    return with_describe_pointer(res, "evchibd")


def cheatsheet():
    return "evchibd: Asymptotic-independence diagnostic χ̄(u)"
