# morie.fn -- function file (rootcoder007/morie)
"""Dantzig's simplex method for a linear programme in standard form.

Source CONSULTED: Dantzig, G. B. (1947), the simplex method, first
circulated as a US Air Force report and published in Koopmans (ed.),
*Activity Analysis of Production and Allocation* (1951), and set out at
length in Dantzig (1963), *Linear Programming and Extensions*.  None of
these could be retrieved here, so what is implemented is the standard
published tableau form of the method for

    maximise   c'x     subject to  A x <= b,  x >= 0,  b >= 0,

for which the slack basis is immediately feasible and a single phase
suffices.  Requiring b >= 0 is a real restriction, stated rather than
papered over: with a negative right-hand side the slack basis is
infeasible and a phase-1 problem is needed, which this function does
not build.

The entering and leaving variables are chosen by BLAND'S RULE (Bland,
R. G. (1977), "New finite pivoting rules for the simplex method",
*Mathematics of Operations Research* 2(2):103-107): among the columns
with positive reduced cost take the smallest index, and among the rows
attaining the minimum ratio take the one whose basic variable has the
smallest index.  Dantzig's own most-positive-reduced-cost rule can
cycle on degenerate problems; Bland's rule cannot, and being a pure
index rule it also makes the Python and R arms pivot identically.

VERIFIED numerically against ``lpSolve::lp`` in the parity harness --
an independent implementation -- on both a non-degenerate and a
degenerate instance.
"""

from ._richresult import RichResult

__all__ = ["simplex_lp"]


def simplex_lp(c, A, b, max_iter=1000, tol=1e-12):
    """Maximise c'x subject to A x <= b, x >= 0, with b >= 0.

    Parameters
    ----------
    c : sequence, length nvar
        Objective coefficients; the objective is MAXIMISED.
    A : sequence of sequences, shape (ncon, nvar)
        Constraint matrix.
    b : sequence, length ncon
        Right-hand side; every entry must be non-negative.
    max_iter : int
        Pivot budget.  Bland's rule guarantees termination, so hitting
        this means the problem is far larger than intended.
    tol : float
        Zero tolerance for reduced costs and pivot elements.

    Returns
    -------
    RichResult
        Keys ``status`` ("optimal" or "unbounded"), ``x``,
        ``objective``, ``slack``, ``basis``, ``dual``, ``iterations``,
        ``n_var``, ``n_con``, ``method``.
    """
    c = [float(v) for v in c]
    A = [[float(v) for v in row] for row in A]
    b = [float(v) for v in b]
    nvar = len(c)
    ncon = len(A)
    if len(b) != ncon:
        raise ValueError("b must have one entry per row of A")
    for row in A:
        if len(row) != nvar:
            raise ValueError("every row of A needs one entry per variable")
    for v in b:
        if v < 0.0:
            raise ValueError(
                "every entry of b must be non-negative; a negative "
                "right-hand side needs a phase-1 problem, which this "
                "function does not build")

    total = nvar + ncon
    # tableau rows: [A | I | b], objective row: [-c | 0 | 0]
    T = [A[i] + [1.0 if j == i else 0.0 for j in range(ncon)] + [b[i]]
         for i in range(ncon)]
    z = [-v for v in c] + [0.0] * ncon + [0.0]
    basis = [nvar + i for i in range(ncon)]

    status = "optimal"
    it = 0
    for it in range(1, int(max_iter) + 1):
        enter = -1
        for j in range(total):
            if z[j] < -tol:
                enter = j
                break
        if enter == -1:
            it -= 1
            break
        leave = -1
        best = None
        for i in range(ncon):
            if T[i][enter] > tol:
                ratio = T[i][total] / T[i][enter]
                if (best is None or ratio < best - 1e-12
                        or (abs(ratio - best) <= 1e-12
                            and basis[i] < basis[leave])):
                    best = ratio if best is None or ratio < best else best
                    leave = i
        if leave == -1:
            status = "unbounded"
            break
        piv = T[leave][enter]
        T[leave] = [v / piv for v in T[leave]]
        for i in range(ncon):
            if i == leave:
                continue
            f = T[i][enter]
            if f != 0.0:
                T[i] = [T[i][k] - f * T[leave][k] for k in range(total + 1)]
        f = z[enter]
        if f != 0.0:
            z = [z[k] - f * T[leave][k] for k in range(total + 1)]
        basis[leave] = enter
    else:
        raise ValueError("pivot budget exhausted; Bland's rule should have "
                         "terminated, so the problem is larger than max_iter")

    sol = [0.0] * total
    for i in range(ncon):
        sol[basis[i]] = T[i][total]
    x = sol[:nvar]
    slack = sol[nvar:]
    return RichResult(
        payload={
            "status": status,
            "x": x,
            "objective": sum(c[j] * x[j] for j in range(nvar)),
            "slack": slack,
            "basis": list(basis),
            "dual": [z[nvar + i] for i in range(ncon)],
            "iterations": it,
            "n_var": nvar,
            "n_con": ncon,
            "method": "Dantzig simplex, tableau form, Bland's rule",
        }
    )


def cheatsheet():
    return "smplxs: Simplex method LP"
