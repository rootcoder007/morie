# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Fully corrective Frank-Wolfe.

Holloway (1974), "An extension of the Frank and Wolfe method of
feasible directions", Mathematical Programming 6(1):14-27,
doi:10.1007/BF01580219.  The step-size rule of plain Frank-Wolfe is
replaced by a full re-optimisation over the convex hull of every
vertex found so far:

    S_t = S_{t-1} + { argmin_v <grad f(x_t), v> },
    x_{t+1} = argmin { f(x) : x in conv(S_t) }.

The inner minimisation here is a cycle of exact line searches between
the current point and each active vertex (golden section on gamma in
[0, 1]), repeated a fixed number of rounds so both language arms take
identical steps.  Holloway's point is that the extra work per
iteration buys a strictly better iterate than the 2/(t+2) rule.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["fully_corrective_fw"]

_INVPHI = (math.sqrt(5.0) - 1.0) / 2.0


def _line_search(f, x, v, iters=60):
    lo, hi = 0.0, 1.0
    pt = lambda g: [(1.0 - g) * x[j] + g * v[j] for j in range(len(x))]
    c = hi - _INVPHI * (hi - lo)
    d = lo + _INVPHI * (hi - lo)
    fc = f(pt(c))
    fd = f(pt(d))
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _INVPHI * (hi - lo)
            fc = f(pt(c))
        else:
            lo, c, fc = c, d, fd
            d = lo + _INVPHI * (hi - lo)
            fd = f(pt(d))
    return pt((lo + hi) / 2.0)


def fully_corrective_fw(f, grad_f, domain, x0, steps=10, rounds=3):
    """Frank-Wolfe with a full re-optimisation over the active vertex set."""
    V = core.mat(domain)
    if len(V) == 0:
        raise ValueError("fully_corrective_fw: domain has no vertices")
    d = len(V[0])
    x = core.vec(x0)
    if len(x) != d:
        raise ValueError("fully_corrective_fw: x0 and the vertices have different dimensions")
    if not callable(f) or not callable(grad_f):
        raise ValueError("fully_corrective_fw: f and grad_f must be callable")
    ns = int(steps)
    if ns < 1:
        raise ValueError("fully_corrective_fw: steps must be at least 1")
    active = []
    path = [float(f(x))]
    gap = float("inf")
    for _ in range(ns):
        g = core.vec(grad_f(x))
        best = 0
        bv = None
        for i in range(len(V)):
            s = 0.0
            for j in range(d):
                s += V[i][j] * g[j]
            if bv is None or s < bv:
                bv = s
                best = i
        gx = 0.0
        for j in range(d):
            gx += g[j] * x[j]
        gap = gx - bv
        if best not in active:
            active.append(best)
        for _ in range(int(rounds)):
            for i in active:
                x = _line_search(f, x, V[i])
        path.append(float(f(x)))
    return RichResult(
        title="Fully corrective Frank-Wolfe",
        summary_lines=[("steps", ns), ("active", len(active))],
        payload={
            "estimate": float(f(x)),
            "x": x,
            "f_path": path,
            "gap": gap,
            "n_active": len(active),
            "n": d,
            "method": "vertex addition plus re-optimisation over conv(S), Holloway (1974)",
        },
    )


def cheatsheet():
    return "fwlfwd: fully corrective Frank-Wolfe"
