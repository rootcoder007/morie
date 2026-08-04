# morie.fn -- function file (rootcoder007/morie)
"""Moments of a linear rank statistic."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lrankmom", "gibbons_linrank_moments"]


def lrankmom(a, m):
    """Moments of a linear rank statistic.

    E(T_N) = m * abar,  Var(T_N) = m n / (N(N-1)) * sum (a_i - abar)^2.

    Moments of the linear rank statistic T_N = sum over the m
    treatment ranks of the scores a_i, when the m ranks are a simple
    random sample without replacement from the N = m + n scores.  Both
    theorems come out of sampling without replacement, which is why the
    variance carries the finite-population factor rather than being
    m * Var(a).

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Moments of a linear rank statistic", payload=_c.lrankmom(a=a, m=m))


gibbons_linrank_moments = lrankmom


def cheatsheet():
    return "gb731: Moments of a linear rank statistic"
