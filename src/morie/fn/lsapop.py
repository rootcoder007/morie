"""Linear sum assignment by the Hungarian method with potentials (shortest augmenting paths).

Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics
Quarterly 2, 83-97. Munkres, J. (1957). Algorithms for the assignment and transportation
problems. Journal of SIAM 5, 32-38. The O(n^2 m) shortest-augmenting-path form with dual
potentials follows Jonker and Volgenant (1987), Computing 38, 325-340.
"""

import math

from ._richresult import RichResult

__all__ = ["linear_assignment"]


def linear_assignment(cost, maximize=False):
    r"""Assign each row to a distinct column minimising (or maximising) the total cost.

    For an n x m cost matrix with n <= m every row is assigned; with n > m the
    problem is solved on the transpose. Rows are added one at a time; each
    addition finds a shortest augmenting path in reduced costs
    c_ij - u_i - v_j >= 0 (Dijkstra over columns) and updates the potentials, so
    the result is optimal with dual certificate sum u + sum v = total cost.

    Parameters
    ----------
    cost : list of lists
        Finite costs.
    maximize : bool
        Maximise instead (costs are negated).

    Returns
    -------
    RichResult
        Keys: rows, cols (matched index pairs, rows ascending), total, row_potential, col_potential.

    References
    ----------
    Kuhn, H. W. (1955). Naval Research Logistics Quarterly 2, 83-97.
    Jonker, R. and Volgenant, A. (1987). Computing 38, 325-340.

    Examples
    --------
    >>> linear_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])["total"]
    5.0
    """
    C = [[float(v) for v in row] for row in cost]
    if not C or not C[0] or any(len(r) != len(C[0]) for r in C):
        raise ValueError("cost must be a non-empty rectangular matrix")
    if any(not math.isfinite(v) for r in C for v in r):
        raise ValueError("costs must be finite")
    sign = -1.0 if maximize else 1.0
    tr = len(C) > len(C[0])
    A = [list(r) for r in zip(*C)] if tr else C
    A = [[sign * v for v in r] for r in A]
    n, m = len(A), len(A[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)  # p[j]: row (1-based) matched to column j; column 0 is the virtual start
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = A[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j])
    if tr:
        pairs = sorted((c, r) for r, c in pairs)
    rows = [a for a, _ in pairs]
    cols = [b for _, b in pairs]
    total = 0.0
    for a, b in pairs:
        total += C[a][b]
    up, vp = [sign * x for x in u[1:]], [sign * x for x in v[1:]]
    return RichResult(
        title="Linear assignment",
        summary_lines=[("total", total)],
        payload={
            "rows": rows,
            "cols": cols,
            "total": total,
            "row_potential": vp if tr else up,
            "col_potential": up if tr else vp,
        },
    )


def cheatsheet():
    return "lsapop: Hungarian linear sum assignment with dual potentials (Kuhn 1955; Jonker and Volgenant 1987)"
