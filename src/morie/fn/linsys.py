"""Solve a linear equation system Ax = b: consistency, exact and least-squares solutions."""

import math

from ._richresult import RichResult

__all__ = ["solve_linear_system"]


def _rank(rows, tol):
    """Rank by Gaussian elimination with complete pivoting; pivots below tol * largest |a_ij| count as zero."""
    a = [list(r) for r in rows]
    n, m = len(a), len(a[0])
    scale = max((abs(v) for r in a for v in r), default=0.0)
    if scale == 0:
        return 0
    rk = 0
    for k in range(min(n, m)):
        p, q, big = k, k, 0.0
        for i in range(k, n):
            for j in range(k, m):
                if abs(a[i][j]) > big:
                    p, q, big = i, j, abs(a[i][j])
        if big <= tol * scale:
            break
        a[k], a[p] = a[p], a[k]
        for r in a:
            r[k], r[q] = r[q], r[k]
        for i in range(k + 1, n):
            f = a[i][k] / a[k][k]
            for j in range(k, m):
                a[i][j] -= f * a[k][j]
        rk += 1
    return rk


def _householder_ls(rows, y):
    """Least-squares coefficients and residual sum of squares by Householder QR (full column rank)."""
    r = [list(map(float, v)) for v in rows]
    qty = [float(v) for v in y]
    n, m = len(r), len(r[0])
    for k in range(m):
        nrm = math.sqrt(sum(r[i][k] ** 2 for i in range(k, n)))
        if nrm == 0:
            raise ValueError("design matrix is rank deficient")
        alpha = -nrm if r[k][k] >= 0 else nrm
        v = [r[i][k] for i in range(k, n)]
        v[0] -= alpha
        vv = sum(t * t for t in v)
        for j in range(k, m):
            f = 2 * sum(v[i] * r[k + i][j] for i in range(n - k)) / vv
            for i in range(n - k):
                r[k + i][j] -= f * v[i]
        f = 2 * sum(v[i] * qty[k + i] for i in range(n - k)) / vv
        for i in range(n - k):
            qty[k + i] -= f * v[i]
    dmax = max(abs(r[k][k]) for k in range(m))
    if min(abs(r[k][k]) for k in range(m)) <= dmax * max(n, m) * 2.220446049250313e-16:
        raise ValueError("design matrix is rank deficient")
    coef = [0.0] * m
    for k in range(m - 1, -1, -1):
        coef[k] = (qty[k] - sum(r[k][j] * coef[j] for j in range(k + 1, m))) / r[k][k]
    return coef, sum(t * t for t in qty[m:])


def solve_linear_system(A, b, tol=1e-7):
    r"""Classify and solve :math:`Ax = b` by the ranks of :math:`A` and :math:`(A, b)`.

    Hedderich, Sachs & Reynarowych (2023, eqs 2.47-2.50): the system is
    consistent iff :math:`rg(A, b) = rg(A)`; with full column rank
    :math:`rg(A_{n \times m}) = m` the solution is unique (:math:`A^{-1}b`
    when square) and otherwise, for n > m, the least-squares solution
    :math:`(A'A)^{-1}A'b` (computed by Householder QR, not the normal
    equations). A rank-deficient A has no unique solution (``x`` is None).

    Parameters
    ----------
    A : n x m nested sequence
    b : sequence of n floats
    tol : float
        Relative pivot tolerance for the ranks.

    Returns
    -------
    RichResult
        ``x``, ``rank_A``, ``rank_Ab``, ``consistent``, ``kind``
        ("unique", "least-squares" or "rank-deficient"), ``rss``.

    References
    ----------
    Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations (4th ed.),
    sec. 5.3.
    """
    rows = [[float(v) for v in r] for r in A]
    bb = [float(v) for v in b]
    n, m = len(rows), len(rows[0])
    if n != len(bb) or any(len(r) != m for r in rows):
        raise ValueError("A must be n x m and b of length n")
    ra = _rank(rows, tol)
    rab = _rank([r + [v] for r, v in zip(rows, bb)], tol)
    x, rss = None, None
    if ra == m:
        x, rss = _householder_ls(rows, bb)
        kind = "unique" if rab == ra else "least-squares"
    else:
        kind = "rank-deficient"
    return RichResult(
        title="Linear equation system",
        summary_lines=[("kind", kind), ("rank_A", ra)],
        payload={"x": x, "rank_A": ra, "rank_Ab": rab, "consistent": ra == rab, "kind": kind, "rss": rss},
    )


def cheatsheet():
    return "linsys: consistent iff rg(A, b) = rg(A); full column rank: A^-1 b or (A'A)^-1 A'b by QR"
