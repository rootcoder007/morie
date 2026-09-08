# morie.fn -- function file (rootcoder007/morie)
"""Tail dependence coefficient χ for two series.

Implements sec. 8.4, p. 163 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_chi_tail_dependence"]


def evt_chi_tail_dependence(x, y, u=0.95):
    """chi = lim_{u->1} P(F_Y(Y) > u | F_X(X) > u) (Coles 2001
    sec. 8.4, p. 163), estimated by the observed conditional
    proportion at quantile level ``u``."""
    xs = _ev._flat(x)
    ys = _ev._flat(y)
    n = len(xs)
    if n != len(ys) or n < 4:
        raise ValueError("x and y must be equal-length, n >= 4")
    from .chiDep import _ranks01
    rx = _ranks01(xs)
    ry = _ranks01(ys)
    nx = sum(1 for a in rx if a > u)
    if nx == 0:
        chi = 0.0
    else:
        chi = sum(1 for a, b in zip(rx, ry)
                  if a > u and b > u) / nx
    res = RichResult(payload={"chi": float(chi), "u": float(u), "n": n,
                              "method": "conditional exceedance chi (Coles 2001 p. 163)"})
    return with_describe_pointer(res, "evchitd")


def cheatsheet():
    return "evchitd: Tail dependence coefficient χ for two series"
