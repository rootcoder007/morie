# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Linear programme by the primal simplex method.

The GLPK reference manual (Makhorin, GNU Linear Programming Kit)
documents glp_simplex as a primal/dual simplex over

    minimise c'x   subject to   A x <= b,  x >= 0.

This is that problem solved by the textbook primal simplex on the
slack tableau, with Bland's rule (Bland 1977, Math. of Operations
Research 2(2):103-107, doi:10.1287/moor.2.2.103) for entering and
leaving variables so the iteration cannot cycle.  Only origin-feasible
problems (b >= 0) are accepted; a negative right-hand side needs a
phase-one problem and is refused rather than silently mishandled.
The final objective row carries the simplex multipliers (negated, since
the row is updated by subtraction), so strong duality c'x* = b'y* is
available as a check.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["glpk_lp"]

_EPS = 1e-12


def glpk_lp(c, A, b, max_iter=200):
    """Minimise c'x subject to A x <= b, x >= 0."""
    cv = core.vec(c)
    M = core.mat(A)
    bv = core.vec(b)
    m = len(M)
    if m == 0:
        raise ValueError("glpk_lp: A has no rows")
    n = len(cv)
    if n == 0:
        raise ValueError("glpk_lp: c is empty")
    if len(bv) != m:
        raise ValueError("glpk_lp: A and b have different row counts")
    for r in M:
        if len(r) != n:
            raise ValueError("glpk_lp: A and c have different column counts")
    for v in bv:
        if v < 0:
            raise ValueError("glpk_lp: needs b >= 0 (origin-feasible); phase one is not implemented")
    T = [[M[i][j] for j in range(n)] + [1.0 if k == i else 0.0 for k in range(m)] + [bv[i]] for i in range(m)]
    z = [cv[j] for j in range(n)] + [0.0] * m + [0.0]
    basis = [n + i for i in range(m)]
    it = 0
    status = "optimal"
    while it < int(max_iter):
        enter = -1
        for j in range(n + m):
            if z[j] < -_EPS:
                enter = j
                break
        if enter < 0:
            break
        leave = -1
        best = None
        for i in range(m):
            if T[i][enter] > _EPS:
                ratio = T[i][-1] / T[i][enter]
                if best is None or ratio < best - _EPS or (abs(ratio - best) <= _EPS and basis[i] < basis[leave]):
                    best = ratio
                    leave = i
        if leave < 0:
            status = "unbounded"
            break
        piv = T[leave][enter]
        T[leave] = [v / piv for v in T[leave]]
        for i in range(m):
            if i != leave and abs(T[i][enter]) > 0.0:
                fac = T[i][enter]
                T[i] = [T[i][k] - fac * T[leave][k] for k in range(n + m + 1)]
        fac = z[enter]
        z = [z[k] - fac * T[leave][k] for k in range(n + m + 1)]
        basis[leave] = enter
        it += 1
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][-1]
    obj = 0.0
    for j in range(n):
        obj += cv[j] * x[j]
    y = [-z[n + i] for i in range(m)]
    dual = 0.0
    for i in range(m):
        dual += bv[i] * y[i]
    return RichResult(
        title="Linear programme (primal simplex)",
        summary_lines=[("variables", n), ("constraints", m), ("iterations", it)],
        payload={
            "estimate": obj,
            "x": x,
            "objective": obj,
            "dual": y,
            "dual_objective": dual,
            "iterations": it,
            "status": status,
            "n": n,
            "method": "primal simplex on the slack tableau with Bland's rule",
        },
    )


def cheatsheet():
    return "glpopt: linear programme by the simplex method"


# compact alias per ledger/NAMING.md
glpklp = glpk_lp
