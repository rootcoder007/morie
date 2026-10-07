# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Linear programme by the two-phase primal simplex method.

The GLPK reference manual (Makhorin, GNU Linear Programming Kit)
documents glp_simplex as a primal/dual simplex over

    minimise c'x   subject to   A x <= b,  x >= 0.

This is that problem solved by the textbook primal simplex on the
slack tableau, with Bland's rule (Bland 1977, Math. of Operations
Research 2(2):103-107, doi:10.1287/moor.2.2.103) for entering and
leaving variables so the iteration cannot cycle.  A row with a negative
right-hand side is multiplied by -1 and given an artificial variable;
phase one (Dantzig, Orden and Wolfe 1955) minimises the sum of the
artificials, and a positive minimum means no x satisfies the
constraints (status "infeasible").  Artificials left basic at zero are
pivoted out where the row allows, then phase two minimises c'x.  The
final objective row carries the simplex multipliers (negated, since the
row is updated by subtraction): y = -z on the slack columns, for
flipped rows too since their slack column is flipped with them, so
strong duality c'x* = b'y* is available as a check.  Hitting max_iter
is reported as "iteration_limit", never as an optimum.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["glpk_lp"]

_EPS = 1e-12


def _simplex(T, z, basis, allowed, max_iter, it):
    """Bland's-rule pivots until no allowed column has a negative cost."""
    m = len(T)
    width = len(z)
    status = "optimal"
    while True:
        enter = -1
        for j in allowed:
            if z[j] < -_EPS:
                enter = j
                break
        if enter < 0:
            break
        if it >= max_iter:
            status = "iteration_limit"
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
            if i != leave and T[i][enter] != 0.0:
                fac = T[i][enter]
                T[i] = [T[i][k] - fac * T[leave][k] for k in range(width)]
        fac = z[enter]
        z = [z[k] - fac * T[leave][k] for k in range(width)]
        basis[leave] = enter
        it += 1
    return T, z, basis, status, it


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
    max_iter = int(max_iter)
    sg = [-1.0 if v < 0 else 1.0 for v in bv]
    flip = [i for i in range(m) if sg[i] < 0]
    k = len(flip)
    art = [n + m + q for q in range(k)]
    T = []
    for i in range(m):
        row = [sg[i] * M[i][j] for j in range(n)]
        row += [sg[i] if q == i else 0.0 for q in range(m)]
        row += [1.0 if flip[q] == i else 0.0 for q in range(k)]
        row.append(sg[i] * bv[i])
        T.append(row)
    width = n + m + k + 1
    basis = [n + i for i in range(m)]
    for q, i in enumerate(flip):
        basis[i] = art[q]
    it = 0
    status = "optimal"
    if k > 0:
        w = [0.0] * (n + m) + [1.0] * k + [0.0]
        for i in flip:
            w = [w[c_] - T[i][c_] for c_ in range(width)]
        for a_ in art:
            w[a_] = 0.0
        T, w, basis, st1, it = _simplex(T, w, basis, list(range(n + m)), max_iter, it)
        if st1 == "iteration_limit":
            status = "iteration_limit"
        elif -w[-1] > 1e-9:
            status = "infeasible"
        for i in range(m):
            if basis[i] in art:
                j = next((j for j in range(n + m) if abs(T[i][j]) > 1e-12), None)
                if j is None:
                    continue
                piv = T[i][j]
                T[i] = [v / piv for v in T[i]]
                for r in range(m):
                    if r != i and T[r][j] != 0.0:
                        fac = T[r][j]
                        T[r] = [T[r][c_] - fac * T[i][c_] for c_ in range(width)]
                basis[i] = j
    z = None
    if status == "optimal":
        cost = list(cv) + [0.0] * (m + k) + [0.0]
        z = list(cost)
        for i in range(m):
            cb = cost[basis[i]]
            if cb != 0.0:
                z = [z[c_] - cb * T[i][c_] for c_ in range(width)]
        T, z, basis, status, it = _simplex(T, z, basis, list(range(n + m)), max_iter, it)
    method = "two-phase primal simplex with Bland's rule"
    if status != "optimal":
        nan = float("nan")
        return RichResult(
            title="Linear programme (two-phase simplex)",
            summary_lines=[("variables", n), ("constraints", m), ("status", status)],
            payload={
                "estimate": nan,
                "x": [nan] * n,
                "objective": nan,
                "dual": [nan] * m,
                "dual_objective": nan,
                "iterations": it,
                "status": status,
                "n": n,
                "method": method,
            },
        )
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
        title="Linear programme (two-phase simplex)",
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
            "method": method,
        },
    )


def cheatsheet():
    return "glpopt: linear programme by the simplex method"


# compact alias per ledger/NAMING.md
glpklp = glpk_lp
