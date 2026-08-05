# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Huber weight function.

Huber (1964), "Robust estimation of a location parameter", Ann. Math.
Statist. 35(1):73-101, doi:10.1214/aoms/1177703732.  The weight is the
ratio psi(r)/r of Huber's score function to the residual,

    w(r) = 1              if |r| <= k,
           k / |r|        if |r| >  k,

so an IRLS step with these weights solves the Huber M-estimating
equation.  Weights are scale-equivariant: w(cr; ck) = w(r; k).
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["huber_weight"]


def huber_weight(y, k=1.345):
    """Huber weights of a residual vector.

    Parameters
    ----------
    y : array-like
        Residuals (already divided by a scale estimate if one is used).
    k : float
        Tuning constant; 1.345 gives 95% efficiency at the normal.
    """
    r = core.vec(y)
    if len(r) == 0:
        raise ValueError("huber_weight: y is empty")
    kv = float(k)
    if kv <= 0:
        raise ValueError("huber_weight: k must be positive")
    w = [1.0 if abs(v) <= kv else kv / abs(v) for v in r]
    psi = [max(-kv, min(kv, v)) for v in r]
    ndown = sum(1 for v in r if abs(v) > kv)
    tot = 0.0
    for v in w:
        tot += v
    return RichResult(
        title="Huber weights",
        summary_lines=[("n", len(r)), ("k", kv)],
        payload={
            "estimate": tot / len(r),
            "weights": w,
            "psi": psi,
            "n_downweighted": ndown,
            "k": kv,
            "n": len(r),
            "method": "w(r) = min(1, k/|r|), Huber (1964)",
        },
    )


def cheatsheet():
    return "huberw: Huber psi-weight function"


# compact alias per ledger/NAMING.md
huberweight = huber_weight
