# morie.fn -- function file (rootcoder007/morie)
"""Expected hitting time of a simple random walk on a graph."""

from __future__ import annotations

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["hitting_time"]


def hitting_time(G, start=None, target=0):
    """Expected steps for a simple random walk to reach ``target``.

    Formula: with ``P_ij = w_ij / sum_k w_ik`` the walk's transition
    matrix, the hitting times ``H(i, target)`` solve

        ``H(target, target) = 0``,
        ``H(i, target) = 1 + sum_j P_ij H(j, target)``   for ``i != target``,

    a linear system of size ``n - 1``, solved exactly rather than by
    iterating the walk.  Every vertex from which ``target`` is
    unreachable has infinite hitting time and is reported as such
    instead of being dropped or given a large finite number; the linear
    system is solved only over the vertices that can reach it.

    Hitting time is not symmetric -- ``H(i,j)`` and ``H(j,i)`` differ in
    general -- which is why the commute time ``H(i,j) + H(j,i)`` is the
    quantity with metric behaviour.  Weights are read as edge weights,
    so an unweighted graph is the 0/1 case.

    Parameters
    ----------
    G : array-like
        ``n x n`` weight matrix; non-negative, zero meaning no edge.
    start : int, optional
        Starting vertex.  All vertices are returned regardless; this
        just selects the scalar reported as ``estimate``.
    target : int
        Vertex to be hit.

    Returns
    -------
    RichResult
        ``estimate`` (``H(start, target)``), ``hitting`` (all of them),
        ``target``, ``start``, ``n``, ``method``.

    References
    ----------
    Lovasz (1996), Random walks on graphs: a survey, in Combinatorics,
    Paul Erdos is Eighty, vol. 2, Bolyai Society Mathematical Studies,
    pp. 353-398.  The PDF on Lovasz's ELTE page could not be fetched
    from this host (expired TLS certificate on web.cs.elte.hu, and the
    plain curl returned nothing), so this is the standard first-step
    recurrence rather than a quoted equation.  It is anchored in the
    test harness on the cycle ``C_n``, for which the classical closed
    form ``H(i, j) = d (n - d)`` with ``d`` the cyclic distance holds
    exactly and is independent of this code.
    """
    W = T.mat(G)
    n = len(W)
    if n < 2 or any(len(r) != n for r in W):
        raise ValueError("G must be a square weight matrix with n >= 2")
    if any(v < 0 for row in W for v in row):
        raise ValueError("weights must be non-negative")
    target = int(target)
    if not 0 <= target < n:
        raise ValueError("target out of range")
    if start is None:
        start = 0 if target != 0 else 1
    start = int(start)
    if not 0 <= start < n:
        raise ValueError("start out of range")
    # vertices that can reach the target, found by BFS on reversed edges
    reach = [False] * n
    reach[target] = True
    queue = [target]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for u in range(n):
            if not reach[u] and W[u][v] > 0:
                reach[u] = True
                queue.append(u)
    idx = [i for i in range(n) if reach[i] and i != target]
    pos = {v: k for k, v in enumerate(idx)}
    k = len(idx)
    H = [float("inf")] * n
    H[target] = 0.0
    if k:
        A = [[0.0] * k for _ in range(k)]
        b = [1.0] * k
        for r, i in enumerate(idx):
            deg = sum(W[i])
            if deg <= 0:
                raise ValueError(f"vertex {i} reaches the target but has no outgoing weight")
            A[r][r] = 1.0
            for j in range(n):
                if W[i][j] <= 0 or j == target:
                    continue
                if j in pos:
                    A[r][pos[j]] -= W[i][j] / deg
        sol = _gauss(A, b)
        for r, i in enumerate(idx):
            H[i] = sol[r]
    return RichResult(
        payload={
            "estimate": float(H[start]),
            "hitting": H,
            "target": int(target),
            "start": int(start),
            "n": int(n),
            "method": "Expected hitting time of a simple random walk",
        }
    )


def _gauss(A, b):
    """Solve ``A z = b`` by Gauss-Jordan with partial pivoting."""
    k = len(A)
    aug = [A[i][:] + [b[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-300:
            raise ValueError("singular hitting-time system")
        aug[c], aug[piv] = aug[piv], aug[c]
        d = aug[c][c]
        aug[c] = [v / d for v in aug[c]]
        for r in range(k):
            if r == c:
                continue
            f = aug[r][c]
            if f != 0.0:
                aug[r] = [aug[r][j] - f * aug[c][j] for j in range(k + 1)]
    return [row[k] for row in aug]


def cheatsheet():
    return "hitting_time(G, start, target): H(i,t) = 1 + sum_j P_ij H(j,t)."


# compact alias per ledger/NAMING.md
hittingtime = hitting_time
