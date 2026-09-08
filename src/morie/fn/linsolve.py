# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Solving a linear equation system, with the solvability criterion.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 2.4.6, printed page 50 (PDF page 84), equations (2.47) to (2.50)::

    a11 x1 + a12 x2 + ... + a1m xm = b1
      ...                                                          (2.47)
    an1 x1 + an2 x2 + ... + anm xm = bn

    rg(A, b) = rg(A)          <=> the system has a solution         (2.48)

    x = A^-1 b                is a unique solution                  (2.49)
      (case 1: A square, rg(A_(m,m)) = m)

    x = (A'A)^-1 A' b         is the OLS solution                   (2.50)
      (case 2: A has full column rank rg(A_(n,m)) = m < n)

(2.47) is the system itself, (2.48) the Rouche-Capelli consistency
criterion, and (2.49)/(2.50) the two solved cases the book distinguishes.
All three are one method and live in this one function.
"""

from __future__ import annotations

from .mrank import _mm, _rank, _rows, _t
from ._richresult import RichResult

__all__ = ["linsolve"]


def _solve(a, b):
    """Gauss-Jordan with partial pivoting; ``a`` square, ``b`` a column."""
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda i: abs(m[i][c]))
        if m[piv][c] == 0.0:
            raise ValueError("matrix is singular")
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        for j in range(c, n + 1):
            m[c][j] /= pv
        for i in range(n):
            if i != c and m[i][c] != 0.0:
                f = m[i][c]
                for j in range(c, n + 1):
                    m[i][j] -= f * m[c][j]
    return [m[i][n] for i in range(n)]


def linsolve(A, b):
    """Solve ``A x = b`` following the case distinction of (2.48)-(2.50).

    Parameters
    ----------
    A : 2-D array-like
        Coefficient matrix, ``n`` by ``m``.
    b : array-like
        Right-hand side, length ``n``.

    Returns
    -------
    RichResult
        Keys: ``consistent`` (the (2.48) decision), ``rank``,
        ``rank_augmented``, ``nrow``, ``ncol``, ``case``, ``solution``,
        ``residual_norm``, ``homogeneous``.  ``case`` is ``"unique"``
        when (2.49) applies, ``"ols"`` when (2.50) applies,
        ``"underdetermined"`` when the solution is not unique, and
        ``"none"`` when the system is inconsistent, in which case
        ``solution`` is ``None``.
    """
    a = _rows(A)
    if len(a) == 0 or len(a[0]) == 0:
        raise ValueError("A must have at least one row and one column")
    n, m = len(a), len(a[0])
    bb = [float(v) for v in (b if hasattr(b, "__len__") else [b])]
    if len(bb) != n:
        raise ValueError("length of b must equal the number of rows of A")
    for v in bb:
        if not (v == v) or v in (float("inf"), float("-inf")):
            raise ValueError("b must be finite")
    rk = _rank(a)[0]
    aug = [a[i][:] + [bb[i]] for i in range(n)]
    rka = _rank(aug)[0]
    consistent = rk == rka
    homogeneous = all(v == 0.0 for v in bb)
    sol = None
    case = "none"
    if consistent:
        if n == m and rk == m:
            sol = _solve(a, bb)
            case = "unique"
        elif rk == m:
            at = _t(a)
            sol = _solve(_mm(at, a), [sum(at[i][k] * bb[k] for k in range(n)) for i in range(m)])
            case = "ols"
        else:
            case = "underdetermined"
    elif rk == m:
        # inconsistent but full column rank: the book's (2.50) still gives
        # the least-squares fit, reported for reference but not a solution
        at = _t(a)
        sol = _solve(_mm(at, a), [sum(at[i][k] * bb[k] for k in range(n)) for i in range(m)])
        case = "none"
    resid = None
    if sol is not None:
        resid = sum((sum(a[i][j] * sol[j] for j in range(m)) - bb[i]) ** 2 for i in range(n)) ** 0.5
    payload = {
        "consistent": consistent,
        "rank": rk,
        "rank_augmented": rka,
        "nrow": n,
        "ncol": m,
        "case": case,
        "solution": sol,
        "residual_norm": resid,
        "homogeneous": homogeneous,
    }
    return RichResult(
        title="Linear equation system (Hedderich eq. 2.47-2.50)",
        summary_lines=[
            ("rg(A, b) = rg(A)", consistent),
            ("rank / augmented rank", (rk, rka)),
            ("case", case),
            ("solution", sol),
        ],
        payload=payload,
    )


def cheatsheet() -> str:
    return "linsolve(A, b): solvability rg(A,b)=rg(A) and the solution -- Hedderich eq. (2.47)-(2.50)."
