# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Frank-Wolfe conditional gradient.

Frank and Wolfe (1956), "An algorithm for quadratic programming",
Naval Research Logistics Quarterly 3(1-2):95-110,
doi:10.1002/nav.3800030109.  Each step solves a LINEAR problem over
the feasible polytope and moves toward its solution:

    s_t = argmin_{v in vertices} <grad f(x_t), v>,
    x_{t+1} = (1 - gamma_t) x_t + gamma_t s_t,   gamma_t = 2/(t + 2).

The iterate stays in the polytope by construction -- no projection is
ever needed, which is the whole appeal of the method.  The quantity
g_t = <grad f(x_t), x_t - s_t> is the Frank-Wolfe duality gap and
upper bounds f(x_t) - f*, so it is reported alongside.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["frank_wolfe"]


def _argmin_vertex(V, g):
    best = 0
    bv = None
    for i in range(len(V)):
        s = 0.0
        for j in range(len(g)):
            s += V[i][j] * g[j]
        if bv is None or s < bv:
            bv = s
            best = i
    return best, bv


def frank_wolfe(f, grad_f, domain, x0, steps=50):
    """Conditional-gradient minimisation over the hull of the given vertices."""
    V = core.mat(domain)
    if len(V) == 0:
        raise ValueError("frank_wolfe: domain has no vertices")
    d = len(V[0])
    x = core.vec(x0)
    if len(x) != d:
        raise ValueError("frank_wolfe: x0 and the vertices have different dimensions")
    if not callable(f) or not callable(grad_f):
        raise ValueError("frank_wolfe: f and grad_f must be callable")
    ns = int(steps)
    if ns < 1:
        raise ValueError("frank_wolfe: steps must be at least 1")
    path = [float(f(x))]
    gap = float("inf")
    for t in range(ns):
        g = core.vec(grad_f(x))
        if len(g) != d:
            raise ValueError("frank_wolfe: gradient has the wrong length")
        i, sv = _argmin_vertex(V, g)
        gx = 0.0
        for j in range(d):
            gx += g[j] * x[j]
        gap = gx - sv
        gamma = 2.0 / (t + 2.0)
        x = [(1.0 - gamma) * x[j] + gamma * V[i][j] for j in range(d)]
        path.append(float(f(x)))
    return RichResult(
        title="Frank-Wolfe",
        summary_lines=[("steps", ns), ("vertices", len(V))],
        payload={
            "estimate": float(f(x)),
            "x": x,
            "f_path": path,
            "gap": gap,
            "steps": ns,
            "n": d,
            "method": "s = argmin_v <grad, v>, gamma_t = 2/(t+2), Frank & Wolfe (1956)",
        },
    )


def cheatsheet():
    return "frwol2: Frank-Wolfe conditional gradient"


# compact alias per ledger/NAMING.md
frankwolfe = frank_wolfe
