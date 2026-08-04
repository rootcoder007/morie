# morie.fn -- function file (rootcoder007/morie)
"""Linear-programming duality certificate."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lpdual", "lp_dual"]


def lpdual(A, b, c, x=None, y=None):
    """Form the dual of a linear program and certify a pair of solutions.

    No linear program is SOLVED here, deliberately.  Duality is a
    certificate, and the useful operation is checking one: given a
    feasible primal x and a feasible dual y with equal objectives, the
    pair is optimal, and no simplex run is needed to know it.  When
    the objectives differ, ``gap`` is the exact distance from
    optimality -- which is more information than a solver's answer
    alone.

    Complementary slackness is reported per constraint and per
    variable rather than as a single flag, because the binding
    constraints are what the certificate is about.

    Formula: primal   max c'x  s.t. A x <= b, x >= 0
             dual     min b'y  s.t. A'y >= c, y >= 0
             weak duality  c'x <= b'y for any feasible pair
             complementary slackness  y_i (b - A x)_i = 0 and
                                      x_j (A'y - c)_j = 0

    Parameters
    ----------
    A : array-like, shape (m, n)
        Constraint matrix.
    b : array-like
        Right-hand side, length m.
    c : array-like
        Objective coefficients, length n.
    x : array-like, optional
        Candidate primal solution.
    y : array-like, optional
        Candidate dual solution.

    Returns
    -------
    RichResult
        ``dual_A`` (A'), ``dual_b`` (c), ``dual_c`` (b),
        ``primal_objective``, ``dual_objective``, ``gap``,
        ``primal_feasible``, ``dual_feasible``, ``optimal``,
        ``slack``, ``surplus``, ``cs_constraint``, ``cs_variable``,
        ``m``, ``n``.

    References
    ----------
    von Neumann, J. (1947), Discussion of a maximum problem,
    unpublished working paper, Institute for Advanced Study, reprinted
    in his Collected Works VI, 89-95 -- the first statement of linear
    programming duality, arrived at from the minimax theorem for
    matrix games.  Gale, Kuhn & Tucker (1951), Linear programming and
    the theory of games, in Activity Analysis of Production and
    Allocation, 317-329, for the first published proof.
    """
    A = C.mat(A)
    m = len(A)
    if m < 1:
        raise ValueError("the constraint matrix must be non-empty")
    n = len(A[0])
    if any(len(r) != n for r in A):
        raise ValueError("the constraint matrix must be rectangular")
    b = C.vec(b)
    c = C.vec(c)
    if len(b) != m:
        raise ValueError("b must have one entry per constraint")
    if len(c) != n:
        raise ValueError("c must have one entry per variable")
    At = C.transpose(A)
    if x is None:
        x = [0.0] * n
    else:
        x = C.vec(x)
        if len(x) != n:
            raise ValueError("x must have one entry per variable")
    if y is None:
        y = [0.0] * m
    else:
        y = C.vec(y)
        if len(y) != m:
            raise ValueError("y must have one entry per constraint")
    Ax = C.matvec(A, x)
    Aty = C.matvec(At, y)
    slack = [b[i] - Ax[i] for i in range(m)]
    surp = [Aty[j] - c[j] for j in range(n)]
    pf = 1.0 if (all(v >= -1e-9 for v in slack)
                 and all(v >= -1e-9 for v in x)) else 0.0
    df = 1.0 if (all(v >= -1e-9 for v in surp)
                 and all(v >= -1e-9 for v in y)) else 0.0
    po = sum(c[j] * x[j] for j in range(n))
    do = sum(b[i] * y[i] for i in range(m))
    gap = do - po
    return RichResult(payload={
        "dual_A": At, "dual_b": c, "dual_c": b,
        "primal_objective": po, "dual_objective": do, "gap": gap,
        "primal_feasible": pf, "dual_feasible": df,
        "optimal": 1.0 if (pf and df and abs(gap) <= 1e-9) else 0.0,
        "slack": slack, "surplus": surp,
        "cs_constraint": [y[i] * slack[i] for i in range(m)],
        "cs_variable": [x[j] * surp[j] for j in range(n)],
        "m": float(m), "n": float(n),
        "method": "LP duality certificate (no LP is solved)"})


lp_dual = lpdual


def cheatsheet():
    return "lpdual: max c'x, Ax<=b -> min b'y, A'y>=c; gap = b'y - c'x"
