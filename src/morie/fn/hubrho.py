# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Huber loss function.

Huber (1964), Ann. Math. Statist. 35(1):73-101,
doi:10.1214/aoms/1177703732, equation (1.4) with the convention that
rho is quadratic in the centre and linear in the tails:

    rho(r) = r^2 / 2            if |r| <= k,
             k (|r| - k / 2)    if |r| >  k.

The two branches agree in value AND in slope at |r| = k (both give
k^2/2 and slope k), which is what makes rho convex and C^1; the tests
check both continuities directly.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["huber_loss"]


def huber_loss(r, k=1.345):
    """Huber loss of a residual vector, and its total."""
    v = core.vec(r)
    if len(v) == 0:
        raise ValueError("huber_loss: r is empty")
    kv = float(k)
    if kv <= 0:
        raise ValueError("huber_loss: k must be positive")
    rho = [(x * x / 2.0) if abs(x) <= kv else kv * (abs(x) - kv / 2.0) for x in v]
    psi = [max(-kv, min(kv, x)) for x in v]
    tot = 0.0
    for x in rho:
        tot += x
    return RichResult(
        title="Huber loss",
        summary_lines=[("n", len(v)), ("k", kv)],
        payload={
            "estimate": tot,
            "loss": rho,
            "psi": psi,
            "mean_loss": tot / len(v),
            "k": kv,
            "n": len(v),
            "method": "rho(r) = r^2/2 for |r| <= k else k(|r| - k/2), Huber (1964)",
        },
    )


def cheatsheet():
    return "hubrho: Huber loss function"


# compact alias per ledger/NAMING.md
huberloss = huber_loss
