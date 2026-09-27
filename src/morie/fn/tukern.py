# morie.fn -- function file (rootcoder007/morie)
"""A kernel (or prekernel) point of a TU game by Stearns' transfer scheme."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._tucore import max_surplus, players, worth


def kernel_point(v, x0=None, *, pre: bool = False, tol: float = 1e-10, max_iter: int = 100000) -> DescriptiveResult:
    """A point of the kernel (Davis and Maschler 1965) by Stearns' transfers.

    The maximum surplus of ``i`` over ``j`` at ``x`` is
    ``s_ij(x) = max {v(S) - x(S) : i in S, j not in S}``. An imputation is in
    the kernel when ``s_ij > s_ji`` implies ``x_j = v({j})`` for every pair;
    the prekernel asks ``s_ij = s_ji``. Starting from ``x0`` (default: the
    equal split of ``v(N) - sum v({i})`` on top of the singleton worths),
    each step moves ``min((s_ij - s_ji) / 2, x_j - v({j}))`` (no bound for
    the prekernel) from ``j`` to ``i`` for the pair where that amount is
    largest. Stearns (1968) shows the scheme converges to a kernel point.

    :param v: Worths in binary order, length ``2**n - 1``.
    :param x0: Starting imputation.
    :param pre: Prekernel instead of kernel.
    :param tol: Stop when no pair can transfer more than ``tol``.
    :param max_iter: Maximum number of transfers.
    :return: DescriptiveResult; ``value`` is the allocation, ``extra`` has
        ``surplus`` (matrix of s_ij), ``in_kernel`` (the membership test at
        tolerance ``100 * tol``), ``iterations`` and ``converged``.

    References
    ----------
    Davis, M. and Maschler, M. (1965). The kernel of a cooperative game.
    Naval Research Logistics Quarterly 12, 223-259.

    Stearns, R. E. (1968). Convergent transfer schemes for n-person games.
    Transactions of the American Mathematical Society 134, 449-459.

    Examples
    --------
    >>> r = kernel_point([0, 0, 60, 0, 60, 60, 72])
    >>> [round(a, 6) for a in r.value], r.extra["in_kernel"]
    ([24.0, 24.0, 24.0], True)
    """
    n = players(v)
    single = [worth(v, 1 << i) for i in range(n)]
    if x0 is None:
        share = (worth(v, 2**n - 1) - sum(single)) / n
        x = [s + share for s in single]
    else:
        x = [float(a) for a in x0]
    it, converged = 0, False
    while it < max_iter:
        best, pair = 0.0, None
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = (max_surplus(v, x, i, j) - max_surplus(v, x, j, i)) / 2.0
                step = d if pre else min(d, x[j] - single[j])
                if step > best:
                    best, pair = step, (i, j)
        if pair is None or best <= tol:
            converged = True
            break
        i, j = pair
        x[i] += best
        x[j] -= best
        it += 1
    s = [[max_surplus(v, x, i, j) if i != j else 0.0 for j in range(n)] for i in range(n)]
    ok = all(
        abs(s[i][j] - s[j][i]) <= 100 * tol if pre else s[i][j] <= s[j][i] + 100 * tol or x[j] - single[j] <= 100 * tol
        for i in range(n)
        for j in range(n)
        if i != j
    )
    return DescriptiveResult(
        name="prekernel_point" if pre else "kernel_point",
        value=x,
        extra={"surplus": s, "in_kernel": ok, "iterations": it, "converged": converged},
    )


tukern = kernel_point


def cheatsheet() -> str:
    return "kernel_point(v) -> kernel / prekernel point of a TU game (Stearns transfers)"
