# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Barrier interior-point solver for smooth inequality-constrained NLPs.

Waechter and Biegler (2006), "On the implementation of an
interior-point filter line-search algorithm for large-scale nonlinear
programming", Mathematical Programming 106(1):25-57,
doi:10.1007/s10107-004-0559-y -- the IPOPT paper.  The core of that
algorithm is the barrier subproblem

    minimise  f(x) - mu sum_i log(-g_i(x)),   g_i(x) <= 0,

solved approximately for a decreasing sequence of mu by Newton steps
with a backtracking line search that keeps the iterate strictly
feasible.  Derivatives are central differences with a fixed step, so
no symbolic gradient is required of the caller and both language arms
follow the same trajectory.  When a difference stencil would step
outside the feasible set the barrier subproblem is declared finished:
the iterate is then within one difference step of the boundary, which
is the accuracy a barrier method offers there anyway.
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


def _shift(x, idx, amt):
    y = list(x)
    for j in range(len(idx)):
        y[idx[j]] += amt[j]
    return y


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
            plus = [_phi(f, cons, _shift(x, [j], [_H]), mu) for j in range(n)]
            minus = [_phi(f, cons, _shift(x, [j], [-_H]), mu) for j in range(n)]
            cross = {}
            for j in range(n):
                for k in range(j + 1, n):
                    for sj in (1.0, -1.0):
                        for sk in (1.0, -1.0):
                            cross[(j, k, sj, sk)] = _phi(f, cons, _shift(x, [j, k], [sj * _H, sk * _H]), mu)
            vals = [base] + plus + minus + list(cross.values())
            bad = False
            for v in vals:
                if not (v == v) or v == float("inf") or v == float("-inf"):
                    bad = True
            if bad:
                break
            g = [(plus[j] - minus[j]) / (2.0 * _H) for j in range(n)]
            H = [[0.0] * n for _ in range(n)]
            for j in range(n):
                H[j][j] = (plus[j] - 2.0 * base + minus[j]) / (_H * _H) + 1e-8
                for k in range(j + 1, n):
                    v = (cross[(j, k, 1.0, 1.0)] - cross[(j, k, 1.0, -1.0)] - cross[(j, k, -1.0, 1.0)] + cross[(j, k, -1.0, -1.0)]) / (4.0 * _H * _H)
                    H[j][k] = v
                    H[k][j] = v
            try:
                step = core.cholsolve(H, [-v for v in g])
            except Exception:
                step = [-v for v in g]
            a = 1.0
            moved = False
            for _ in range(60):
                xn = [x[j] + a * step[j] for j in range(n)]
                fv = _phi(f, cons, xn, mu)
                if fv == fv and fv < base:
                    x = xn
                    moved = True
                    break
                a *= 0.5
            if not moved:
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
