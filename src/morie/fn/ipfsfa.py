# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Barrier interior-point solver for smooth inequality-constrained NLPs.

Waechter and Biegler (2006), "On the implementation of an
interior-point filter line-search algorithm for large-scale nonlinear
programming", Mathematical Programming 106(1):25-57,
doi:10.1007/s10107-004-0559-y -- the IPOPT paper.  The core of that
algorithm is the barrier subproblem

    minimise  f(x) - mu sum_i log(-g_i(x)),   g_i(x) <= 0,

solved approximately for a decreasing sequence of mu with a Newton
step and a backtracking line search that keeps the iterate strictly
feasible.  Derivatives are taken by central differences with a fixed
step, so no symbolic gradient is required of the caller and both
language arms follow the same trajectory.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ipopt_solver"]

_H = 1e-5


def _phi(f, cons, x, mu):
    v = float(f(x))
    for g in cons:
        gv = float(g(x))
        if gv >= 0:
            return float("inf")
        v -= mu * math.log(-gv)
    return v


def ipopt_solver(f, constraints, x0, mu0=1.0, outer=8, inner=30):
    """Minimise f subject to g_i(x) <= 0 by a decreasing-mu barrier."""
    x = core.vec(x0)
    n = len(x)
    if n == 0:
        raise ValueError("ipopt_solver: x0 is empty")
    if not callable(f):
        raise ValueError("ipopt_solver: f must be callable")
    cons = list(constraints)
    for g in cons:
        if not callable(g):
            raise ValueError("ipopt_solver: every constraint must be callable")
        if float(g(x)) >= 0:
            raise ValueError("ipopt_solver: x0 must be strictly feasible")
    mu = float(mu0)
    if mu <= 0:
        raise ValueError("ipopt_solver: mu0 must be positive")
    for _ in range(int(outer)):
        for _ in range(int(inner)):
            base = _phi(f, cons, x, mu)
            g = [0.0] * n
            for j in range(n):
                xp = list(x)
                xm = list(x)
                xp[j] += _H
                xm[j] -= _H
                g[j] = (_phi(f, cons, xp, mu) - _phi(f, cons, xm, mu)) / (2.0 * _H)
            H = [[0.0] * n for _ in range(n)]
            for j in range(n):
                xp = list(x)
                xm = list(x)
                xp[j] += _H
                xm[j] -= _H
                H[j][j] = (_phi(f, cons, xp, mu) - 2.0 * base + _phi(f, cons, xm, mu)) / (_H * _H)
                for k in range(j + 1, n):
                    xpp = list(x)
                    xpm = list(x)
                    xmp = list(x)
                    xmm = list(x)
                    xpp[j] += _H
                    xpp[k] += _H
                    xpm[j] += _H
                    xpm[k] -= _H
                    xmp[j] -= _H
                    xmp[k] += _H
                    xmm[j] -= _H
                    xmm[k] -= _H
                    v = (_phi(f, cons, xpp, mu) - _phi(f, cons, xpm, mu) - _phi(f, cons, xmp, mu) + _phi(f, cons, xmm, mu)) / (4.0 * _H * _H)
                    H[j][k] = v
                    H[k][j] = v
            for j in range(n):
                H[j][j] += 1e-8
            try:
                step = core.cholsolve(H, [-v for v in g])
            except Exception:
                step = [-v for v in g]
            a = 1.0
            ok = False
            for _ in range(60):
                xn = [x[j] + a * step[j] for j in range(n)]
                if _phi(f, cons, xn, mu) < base:
                    x = xn
                    ok = True
                    break
                a *= 0.5
            if not ok:
                break
        mu *= 0.2
    viol = max((float(g(x)) for g in cons), default=-1.0)
    return RichResult(
        title="Barrier interior-point solver",
        summary_lines=[("variables", n), ("constraints", len(cons))],
        payload={
            "estimate": float(f(x)),
            "x": x,
            "objective": float(f(x)),
            "max_violation": viol,
            "mu_final": mu,
            "n": n,
            "method": "decreasing-mu log-barrier with Newton steps, Waechter & Biegler (2006)",
        },
    )


def cheatsheet():
    return "ipfsfa: barrier interior-point NLP solver"


# compact alias per ledger/NAMING.md
ipoptsolver = ipopt_solver
