"""Non-negative least squares, Lawson and Hanson's active-set algorithm."""

import math

from ._richresult import RichResult

__all__ = ["nonnegative_least_squares"]


def _lstsq(cols, b):
    """Least squares on the columns ``cols`` (list of column lists) by Householder QR."""
    m, k = len(b), len(cols)
    a = [[cols[j][i] for j in range(k)] for i in range(m)]
    y = list(b)
    for j in range(k):
        norm = math.sqrt(sum(a[i][j] ** 2 for i in range(j, m)))
        if norm == 0.0:
            continue
        alpha = -norm if a[j][j] >= 0 else norm
        v = [0.0] * m
        v[j] = a[j][j] - alpha
        for i in range(j + 1, m):
            v[i] = a[i][j]
        vv = sum(x * x for x in v[j:])
        if vv == 0.0:
            continue
        for c in range(j, k):
            s = 2.0 * sum(v[i] * a[i][c] for i in range(j, m)) / vv
            for i in range(j, m):
                a[i][c] -= s * v[i]
        s = 2.0 * sum(v[i] * y[i] for i in range(j, m)) / vv
        for i in range(j, m):
            y[i] -= s * v[i]
    x = [0.0] * k
    for j in range(k - 1, -1, -1):
        x[j] = (y[j] - sum(a[j][c] * x[c] for c in range(j + 1, k))) / a[j][j]
    return x


def nonnegative_least_squares(A, b, max_iter=None):
    r"""Minimise :math:`\|Ax - b\|_2` subject to :math:`x \ge 0`.

    The active-set algorithm NNLS of Lawson & Hanson (1974, Ch. 23): move
    the variable with the largest gradient :math:`w = A'(b - Ax)` into the
    passive set, solve the unconstrained problem on the passive columns,
    and step back along the segment whenever a passive coefficient would
    turn non-positive. It stops when every active gradient is non-positive
    (the Kuhn-Tucker conditions).

    Parameters
    ----------
    A : array-like, (m, n)
    b : array-like, (m,)
    max_iter : int, optional
        Outer iterations, default ``3 n``.

    Returns
    -------
    RichResult
        ``x``, ``residual_norm`` (:math:`\|Ax - b\|_2`), ``passive`` (indices
        with :math:`x_j > 0`), ``iterations``.

    References
    ----------
    Lawson, C. L. & Hanson, R. J. (1974). Solving Least Squares Problems.
    Prentice-Hall, Ch. 23.
    """
    a = [[float(v) for v in row] for row in A]
    y = [float(v) for v in b]
    m = len(a)
    if m == 0 or len(y) != m:
        raise ValueError("`A` and `b` must have the same number of rows")
    n = len(a[0])
    cols = [[a[i][j] for i in range(m)] for j in range(n)]
    anorm = max(sum(abs(v) for v in c) for c in cols) or 1.0
    tol = 10.0 * 2.220446049250313e-16 * anorm * max(m, n)
    x = [0.0] * n
    passive = []
    it = 0
    limit = 3 * n if max_iter is None else int(max_iter)

    def grad(xv):
        r = [y[i] - sum(a[i][j] * xv[j] for j in range(n)) for i in range(m)]
        return [sum(cols[j][i] * r[i] for i in range(m)) for j in range(n)]

    w = grad(x)
    while it < limit:
        active = [j for j in range(n) if j not in passive]
        if not active or max(w[j] for j in active) <= tol:
            break
        it += 1
        t = max(active, key=lambda j: w[j])
        passive.append(t)
        while True:
            z = _lstsq([cols[j] for j in passive], y)
            zp = dict(zip(passive, z))
            if all(v > 0 for v in z):
                x = [zp.get(j, 0.0) for j in range(n)]
                break
            alpha = min(x[j] / (x[j] - zp[j]) for j in passive if zp[j] <= 0)
            x = [x[j] + alpha * (zp.get(j, 0.0) - x[j]) for j in range(n)]
            passive = [j for j in passive if x[j] > tol]
            for j in range(n):
                if j not in passive:
                    x[j] = 0.0
            if not passive:
                break
        w = grad(x)
    res = math.sqrt(sum((y[i] - sum(a[i][j] * x[j] for j in range(n))) ** 2 for i in range(m)))
    return RichResult(
        title="Non-negative least squares (Lawson-Hanson)",
        summary_lines=[("residual norm", res), ("passive", len(passive))],
        payload={"x": x, "residual_norm": res, "passive": sorted(passive), "iterations": it},
    )


def cheatsheet():
    return "nnlsq: min ||Ax - b|| subject to x >= 0 (Lawson-Hanson NNLS)"
