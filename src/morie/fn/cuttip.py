# morie.fn -- function file (rootcoder007/morie)
"""Cutting plane method."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["cutting_plane"]

_TOL = 1e-9


def _pivot(T, basis, r, c):
    piv = T[r][c]
    T[r] = [v / piv for v in T[r]]
    for i in range(len(T)):
        if i != r and T[i][c] != 0.0:
            f = T[i][c]
            T[i] = [a - f * b for a, b in zip(T[i], T[r])]
    basis[r] = c


def _primal(T, basis, max_iter):
    """Primal simplex on the tableau (last row = reduced costs of the maximisation, Bland's rule)."""
    m = len(T) - 1
    ncol = len(T[0]) - 1
    for _ in range(max_iter):
        c = next((j for j in range(ncol) if T[m][j] < -_TOL), None)
        if c is None:
            return "optimal"
        rows = [i for i in range(m) if T[i][c] > _TOL]
        if not rows:
            return "unbounded"
        r = min(rows, key=lambda i: (T[i][-1] / T[i][c], basis[i]))
        _pivot(T, basis, r, c)
    return "iteration_limit"


def _dual(T, basis, max_iter):
    """Dual simplex: restore primal feasibility after a cut, keeping the reduced costs non-negative."""
    m = len(T) - 1
    ncol = len(T[0]) - 1
    for _ in range(max_iter):
        r = min(range(m), key=lambda i: (T[i][-1], basis[i]))
        if T[r][-1] >= -_TOL:
            return "optimal"
        cols = [j for j in range(ncol) if T[r][j] < -_TOL]
        if not cols:
            return "infeasible"
        c = min(cols, key=lambda j: (T[m][j] / -T[r][j], j))
        _pivot(T, basis, r, c)
    return "iteration_limit"


def _frac(v):
    return v - math.floor(v)


def cutting_plane(c, A, b, integer_indices=None, max_cuts=200, max_iter=5000):
    r"""Cutting plane method.

    Gomory's cutting-plane algorithm for the (mixed-)integer linear program
    ``max c'x`` subject to ``A x <= b``, ``x >= 0``, ``x_j`` integer for
    ``j`` in ``integer_indices`` (all by default). The LP relaxation is
    solved by the primal simplex method from the slack basis (so ``b >= 0``
    is required); while some integer variable is fractional in the optimal
    tableau, the Gomory mixed-integer cut of the most fractional such row
    (Gomory 1960; slacks count as integer when every variable is integer and
    ``A``, ``b`` are integer, as continuous otherwise) is appended and the dual simplex method
    restores feasibility. Each cut removes the current LP optimum and no
    feasible integer point (Nemhauser and Wolsey 1988, ch. II.4). Decisions
    use a ``1e-9`` tolerance.

    Parameters
    ----------
    c : array-like, shape (n,)
        Objective coefficients (maximised).
    A : array-like, shape (m, n)
        Constraint matrix of ``A x <= b``.
    b : array-like, shape (m,)
        Right-hand sides, non-negative.
    integer_indices : sequence of int, optional
        Integer variables (default: all).
    max_cuts : int
        Cut limit.
    max_iter : int
        Pivot limit per simplex call.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``lp_bound`` (relaxation optimum),
        ``n_cuts``, ``status`` (``optimal``, ``cut_limit``, ``infeasible`` or
        ``unbounded``), ``estimate`` (= objective).

    References
    ----------
    Gomory, R. E. (1958). Outline of an algorithm for integer solutions to linear programs. *Bulletin of
    the American Mathematical Society*, 64(5), 275-278.

    Gomory, R. E. (1960). An algorithm for the mixed integer problem. RAND report RM-2597.

    Nemhauser, G. L. and Wolsey, L. A. (1988). *Integer and Combinatorial Optimization*. Wiley.

    Examples
    --------
    >>> r = cutting_plane([1.0, 1.0], [[-2.0, 2.0], [8.0, 10.0]], [1.0, 13.0])
    >>> [round(v, 9) for v in r["x"]], round(r["objective"], 9), round(r["lp_bound"], 9)
    ([1.0, 0.0], 1.0, 1.625)
    """
    cv = [float(v) for v in (c.tolist() if hasattr(c, "tolist") else c)]
    Am = [[float(v) for v in row] for row in (A.tolist() if hasattr(A, "tolist") else A)]
    bv = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
    m, n = len(Am), len(cv)
    if len(bv) != m or any(len(r) != n for r in Am):
        raise ValueError("A must be m x n with len(b) = m and len(c) = n")
    if min(bv) < 0:
        raise ValueError("b >= 0 is required for the slack starting basis")
    ints = set(range(n)) if integer_indices is None else {int(j) for j in integer_indices}
    if (
        ints == set(range(n))
        and all(float(v).is_integer() for row in Am for v in row)
        and all(float(v).is_integer() for v in bv)
    ):
        # pure integer program with integer data: the slacks b - Ax are integer too (stronger cuts)
        ints |= set(range(n, n + m))
    # tableau: n structural + m slack columns, rhs last; objective row holds -c
    T = [Am[i] + [1.0 if k == i else 0.0 for k in range(m)] + [bv[i]] for i in range(m)]
    T.append([-v for v in cv] + [0.0] * m + [0.0])
    basis = [n + i for i in range(m)]
    status = _primal(T, basis, max_iter)
    if status != "optimal":
        return RichResult(
            payload={"x": None, "objective": None, "lp_bound": None, "n_cuts": 0, "status": status, "estimate": None}
        )
    lp_bound = T[-1][-1]
    cuts = 0
    while True:
        cand = [
            (min(_frac(T[i][-1]), 1.0 - _frac(T[i][-1])), -i, i)
            for i in range(len(T) - 1)
            if basis[i] in ints and _TOL < _frac(T[i][-1]) < 1.0 - _TOL
        ]
        if not cand:
            status = "optimal"
            break
        if cuts >= max_cuts:
            status = "cut_limit"
            break
        r = max(cand)[2]
        f0 = _frac(T[r][-1])
        ncol = len(T[0]) - 1
        g = []
        for j in range(ncol):
            if j in basis:
                g.append(0.0)
                continue
            a = T[r][j]
            if j in ints:
                fj = _frac(a)
                g.append(fj / f0 if fj <= f0 else (1.0 - fj) / (1.0 - f0))
            else:
                g.append(a / f0 if a > 0 else -a / (1.0 - f0))
        # cut sum g_j x_j >= 1 as a new row with slack: -g x + s = -1
        for row in T:
            row.insert(ncol, 0.0)
        T.insert(len(T) - 1, [-v for v in g] + [1.0, -1.0])
        basis.append(ncol)
        cuts += 1
        status = _dual(T, basis, max_iter)
        if status != "optimal":
            break
    x = [0.0] * n
    for i, j in enumerate(basis):
        if j < n:
            x[j] = T[i][-1]
    obj = sum(a * b for a, b in zip(cv, x)) if status in ("optimal", "cut_limit") else None
    return RichResult(
        payload={
            "x": x if obj is not None else None,
            "objective": obj,
            "lp_bound": lp_bound,
            "n_cuts": cuts,
            "status": status,
            "estimate": obj,
            "method": "Gomory mixed-integer cutting planes with dual simplex re-optimisation",
        }
    )


cuttip = cutting_plane


def cheatsheet():
    return "cuttip: Gomory cutting-plane method for (mixed-)integer linear programs"


# compact alias per ledger/NAMING.md
cuttingplane = cutting_plane
