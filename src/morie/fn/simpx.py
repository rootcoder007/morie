"""Simplex method for linear programming."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult


def _pivot(T, basis, row, col):
    """Pivot the tableau on (row, col) and record the new basic variable."""
    piv = T[row, col]
    T[row] /= piv
    for i in range(T.shape[0]):
        if i != row and abs(T[i, col]) > 1e-12:
            T[i] -= T[i, col] * T[row]
    basis[row] = col


def _optimise(T, basis, ncols, max_iter):
    """Run primal simplex on an already-feasible tableau.

    Bland's rule (lowest index among eligible columns) is used for both
    the entering and the leaving choice, which guarantees termination on
    degenerate problems. The previous version used Dantzig's most-negative
    rule with no anti-cycling and simply stopped after a fixed iteration
    count, silently returning whatever the tableau held at that moment.
    """
    for _ in range(max_iter):
        col = -1
        for j in range(ncols):
            if T[-1, j] < -1e-9:
                col = j
                break
        if col < 0:
            return "optimal"
        row = -1
        best = None
        for i in range(T.shape[0] - 1):
            if T[i, col] > 1e-12:
                r = T[i, -1] / T[i, col]
                # tie-break on the lowest basic-variable index (Bland)
                if best is None or r < best - 1e-12 or \
                        (abs(r - best) <= 1e-12 and basis[i] < basis[row]):
                    best, row = r, i
        if row < 0:
            return "unbounded"
        _pivot(T, basis, row, col)
    return "iteration_limit"


def simplex_lp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
) -> DescriptiveResult:
    """Simplex method for linear programming.

    Solves: minimise c^T x subject to A_ub @ x <= b_ub, x >= 0.

    Two-phase simplex. Phase 1 is entered only when some b_i < 0, i.e.
    when the all-slack basis is infeasible; artificial variables are
    driven out before the real objective is touched.

    Three defects are fixed here, each of which returned a wrong answer
    with no error raised:

    1. The solution was read back by scanning for columns that "looked
       basic" -- exactly one non-zero entry, equal to 1. That does not
       identify a basis: it ignores the objective row and lets two
       different variables claim the same row. For
       ``min x1+x2 s.t. x1+x2 <= 1`` no pivot is needed, yet both
       columns looked basic, so it reported ``x = (1, 1)`` with objective
       2.0 -- a point that violates the constraint (1+1 > 1). The true
       optimum is 0 at the origin. The basis is now tracked explicitly
       through every pivot.

    2. A negative b_i was handled by multiplying the row by -1. That
       flips the inequality from <= to >= and changes the feasible set.
       Such rows now get a surplus and an artificial variable and go
       through phase 1, which is what the docstring always claimed.

    3. Dantzig's rule with no anti-cycling could stall on a degenerate
       problem; the loop then expired and returned the current tableau as
       if converged. Bland's rule guarantees termination, and a status is
       returned rather than assumed.

    Parameters
    ----------
    c : ndarray
        Objective coefficients (length n).
    A_ub : ndarray
        Inequality constraint matrix (m x n).
    b_ub : ndarray
        Inequality constraint bounds (length m).

    Returns
    -------
    DescriptiveResult
        ``value`` is the optimal objective; ``extra`` has ``x`` and
        ``status`` ("optimal", "infeasible", "unbounded").
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A_ub, dtype=float)
    b = np.asarray(b_ub, dtype=float)
    m, n = A.shape

    # rows needing a surplus + artificial, because b_i < 0 makes the
    # all-slack basis infeasible
    neg = [i for i in range(m) if b[i] < 0]
    n_art = len(neg)
    width = n + m + n_art + 1

    T = np.zeros((m + 1, width))
    basis = [0] * m
    a_at = 0
    for i in range(m):
        sgn = -1.0 if b[i] < 0 else 1.0
        for j in range(n):
            T[i, j] = sgn * A[i, j]
        T[i, n + i] = sgn              # +1 slack, or -1 surplus
        T[i, -1] = sgn * b[i]          # now >= 0
        if b[i] < 0:
            T[i, n + m + a_at] = 1.0   # artificial
            basis[i] = n + m + a_at
            a_at += 1
        else:
            basis[i] = n + i

    max_iter = 50 * (m + n + n_art) + 100

    if n_art:
        # phase 1: minimise the sum of artificials
        for j in range(n + m, n + m + n_art):
            T[-1, j] = 1.0
        for i in range(m):
            if basis[i] >= n + m:
                T[-1] -= T[i]
        st = _optimise(T, basis, n + m + n_art, max_iter)
        # After pricing out the artificial basis the objective row's RHS
        # holds MINUS the phase-1 objective, so residual infeasibility
        # shows up as T[-1,-1] < 0, not > 0. Getting this sign backwards
        # reported an infeasible programme as solved: "x <= -1 and x >= 3"
        # came back optimal at x = 3.
        if st != "optimal" or -T[-1, -1] > 1e-7:
            return DescriptiveResult(
                name="Simplex LP", value=float("nan"),
                extra={"x": np.zeros(n), "status": "infeasible"})
        # drive any artificial still in the basis out at zero level
        for i in range(m):
            if basis[i] >= n + m:
                for j in range(n + m):
                    if abs(T[i, j]) > 1e-9:
                        _pivot(T, basis, i, j)
                        break
        # clear phase-1 objective, install the real one
        for j in range(width):
            T[-1, j] = 0.0
        for j in range(n):
            T[-1, j] = c[j]
        for i in range(m):
            if basis[i] < n and abs(T[-1, basis[i]]) > 1e-12:
                T[-1] -= T[-1, basis[i]] * T[i]
        # artificials must never re-enter
        for j in range(n + m, n + m + n_art):
            T[-1, j] = 1e30
    else:
        for j in range(n):
            T[-1, j] = c[j]

    status = _optimise(T, basis, n + m, max_iter)

    x = np.zeros(n)
    if status == "optimal":
        for i in range(m):
            if basis[i] < n:
                x[basis[i]] = T[i, -1]

    return DescriptiveResult(
        name="Simplex LP",
        value=float(c @ x) if status == "optimal" else float("nan"),
        extra={"x": x, "status": status},
    )


simpx = simplex_lp


# compact alias per ledger/NAMING.md

simplexlp = simplex_lp
