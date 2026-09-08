# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.10: Word Mover's Distance as an exact transport LP."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_wmd"]


def _solve_transport(a, b, C):
    """Exact min-cost transport by successive shortest paths.

    Returns (flow, potentials_u, potentials_v). Optimality is proved
    afterwards by LP duality in the caller, so a wrong answer raises
    rather than being returned.
    """
    m, n = len(a), len(b)
    F = np.zeros((m, n))
    supply = a.astype(float).copy()
    demand = b.astype(float).copy()
    u = np.zeros(m)
    v = np.zeros(n)
    tol = 1e-12
    guard = (m + 1) * (n + 1) + m + n + 10
    for _ in range(guard):
        if supply.sum() <= tol:
            break
        # Bellman-Ford over the residual graph (nodes: 0..m-1 rows,
        # m..m+n-1 columns) from every row with spare supply.
        INF = float("inf")
        dist = np.full(m + n, INF)
        prev = [None] * (m + n)
        dist[:m] = np.where(supply > tol, 0.0, INF)
        for _it in range(m + n):
            changed = False
            for i in range(m):
                if dist[i] == INF:
                    continue
                for j in range(n):
                    if dist[i] + C[i, j] < dist[m + j] - 1e-15:
                        dist[m + j] = dist[i] + C[i, j]
                        prev[m + j] = ("f", i, j)
                        changed = True
            for j in range(n):
                if dist[m + j] == INF:
                    continue
                for i in range(m):
                    if F[i, j] > tol and \
                            dist[m + j] - C[i, j] < dist[i] - 1e-15:
                        dist[i] = dist[m + j] - C[i, j]
                        prev[i] = ("b", i, j)
                        changed = True
            if not changed:
                break
        cand = [j for j in range(n)
                if demand[j] > tol and dist[m + j] < INF]
        if not cand:
            raise ValueError("the transport problem is infeasible: no "
                             "residual path reaches a column still in "
                             "deficit.")
        j = min(cand, key=lambda j: dist[m + j])
        # Walk the path back, collecting the bottleneck.
        path = []
        node = m + j
        while prev[node] is not None:
            kind, pi, pj = prev[node]
            path.append((kind, pi, pj))
            node = pi if kind == "f" else m + pj
        push = demand[j]
        push = min(push, supply[node])
        for kind, pi, pj in path:
            if kind == "b":
                push = min(push, F[pi, pj])
        if push <= tol:
            raise ValueError("the transport solver stalled on a "
                             "zero-capacity augmenting path.")
        for kind, pi, pj in path:
            F[pi, pj] += push if kind == "f" else -push
        supply[node] -= push
        demand[j] -= push
    else:
        raise ValueError("the transport solver did not converge within "
                         "its iteration guard.")
    if supply.sum() > 1e-9 or demand.sum() > 1e-9:
        raise ValueError("the transport problem was left unbalanced by "
                         "the solver.")
    # Optimality certificate: Bellman-Ford from a virtual source over
    # the residual graph. No relaxation left => no negative cycle =>
    # the flow is optimal, and the distances are a feasible LP dual.
    d = np.zeros(m + n)
    for _it in range(m + n):
        changed = False
        for i in range(m):
            for j in range(n):
                if d[i] + C[i, j] < d[m + j] - 1e-12:
                    d[m + j] = d[i] + C[i, j]
                    changed = True
                if F[i, j] > tol and d[m + j] - C[i, j] < d[i] - 1e-12:
                    d[i] = d[m + j] - C[i, j]
                    changed = True
        if not changed:
            break
    else:
        raise ValueError("the residual graph still has a negative "
                         "cycle; the transport solution is not "
                         "optimal.")
    u = -d[:m]
    v = d[m:]
    return F, u, v


def kamath_ch8_wmd(x_n, y_n, C, F=None):
    r"""WMD = min_F <C, F> s.t. F 1 = f_x, F^T 1 = f_y.

    ``x_n`` and ``y_n`` are the n-gram WEIGHT vectors f_{x^n} and
    f_{y^n} (Eq 8.13 builds them); ``C`` is the cost matrix whose
    entries are the Eq 8.11 distances. Supply ``F`` to score a given
    transport plan -- its marginals are checked and the cost <C, F>
    returned; leave it ``None`` and the LP is solved exactly by
    successive shortest paths, with the answer certified by LP
    duality (primal cost == f_x.u + f_y.v) before it is returned.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.10, printed
    p. 326; Kusner et al. (2015).

    Examples
    --------
    >>> out = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5],
    ...                      [[1.0, 3.0], [4.0, 2.0]])
    >>> round(out["estimate"], 12)     # 0.5*1 + 0.5*2, not 0.5*3+0.5*4
    1.5
    """
    a = np.atleast_1d(np.asarray(x_n, dtype=float))
    b = np.atleast_1d(np.asarray(y_n, dtype=float))
    Cm = np.atleast_2d(np.asarray(C, dtype=float))
    if a.size == 0 or b.size == 0:
        raise ValueError("both weight vectors must be non-empty.")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("n-gram weights are a distribution and cannot "
                         "be negative.")
    if Cm.shape != (a.size, b.size):
        raise ValueError(
            f"C must be {a.size}x{b.size}; got {Cm.shape}.")
    if abs(a.sum() - b.sum()) > 1e-9:
        raise ValueError(
            f"the marginals differ in total mass ({a.sum()} vs "
            f"{b.sum()}); the transport problem is infeasible.")
    if a.sum() <= 0:
        raise ValueError("both weight vectors are all zero.")
    if F is not None:
        Fm = np.atleast_2d(np.asarray(F, dtype=float))
        if Fm.shape != Cm.shape:
            raise ValueError(
                f"F must be {Cm.shape}; got {Fm.shape}.")
        if np.any(Fm < -1e-12):
            raise ValueError("a transport plan cannot carry negative "
                             "flow.")
        if np.max(np.abs(Fm.sum(axis=1) - a)) > 1e-8 or \
                np.max(np.abs(Fm.sum(axis=0) - b)) > 1e-8:
            raise ValueError("F violates the marginal constraints "
                             "F 1 = f_x, F^T 1 = f_y.")
        cost = float((Cm * Fm).sum())
        return RichResult(payload={
            "estimate": cost, "flow": [[float(v) for v in row]
                                       for row in Fm],
            "optimal": False, "n": int(a.size),
            "method": "cost of a supplied transport plan (Kamath "
                      "Eq 8.10)"})
    Fm, u, v = _solve_transport(a, b, Cm)
    cost = float((Cm * Fm).sum())
    dual = float(a @ u + b @ v)
    if abs(cost - dual) > 1e-6 * max(1.0, abs(cost)):
        raise ValueError(
            f"the transport solution failed its duality certificate "
            f"(primal {cost}, dual {dual}); no answer is returned.")
    return RichResult(payload={
        "estimate": cost,
        "flow": [[float(w) for w in row] for row in Fm],
        "dual_objective": dual, "potentials_u": [float(w) for w in u],
        "potentials_v": [float(w) for w in v], "optimal": True,
        "n": int(a.size),
        "method": "Word Mover's Distance, exact transport LP "
                  "(Kamath Eq 8.10)"})


def cheatsheet():
    return "km122: exact min-cost transport <C,F>, duality-certified"


# compact alias per ledger/NAMING.md
kamathch8wmd = kamath_ch8_wmd
