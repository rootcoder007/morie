# morie.fn -- function file (rootcoder007/morie)
"""Inverse-probability weight truncation."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["wtrunc", "truncate_weights"]


def wtrunc(w, q=0.99):
    """Inverse-probability weight truncation.

    w_trunc = min(w, quantile_q(w)).

    Truncating inverse-probability weights trades a little bias for a
    large drop in variance: a single near-zero propensity produces a
    weight that dominates the estimate.  The quantile is the type-7
    (R default) sample quantile.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Inverse-probability weight truncation", payload=_c.wtrunc(w=w, q=q))


truncate_weights = wtrunc


def cheatsheet():
    return "gtruncwt: Inverse-probability weight truncation"
