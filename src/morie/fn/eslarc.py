# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Archetypal analysis (Cutler & Breiman 1994; ESL Sec 14.6.1, eqs 14.75-14.77)."""

from ._richresult import RichResult

__all__ = ["esl_archetypes"]


def _solve(M, v):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(v)
    A = [row[:] + [v[i]] for i, row in enumerate(M)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[piv][k]) < 1e-300:
            return None
        A[k], A[piv] = A[piv], A[k]
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            if f:
                for j in range(k, n + 1):
                    A[i][j] -= f * A[k][j]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (A[k][n] - sum(A[k][j] * x[j] for j in range(k + 1, n))) / A[k][k]
    return x


def _simplex_ls(cols, b, max_iter=500):
    """min ||sum_j x_j cols[j] - b||^2 over the unit simplex (x >= 0, sum x = 1), exact active set.

    Lawson-Hanson on the simplex: the passive set P solves the equality-constrained problem
    through its KKT system; a column enters when its multiplier g_j + nu is negative.
    """
    n = len(cols)
    G = [[sum(a * c for a, c in zip(cols[i], cols[j])) for j in range(n)] for i in range(n)]
    h = [sum(a * c for a, c in zip(cols[i], b)) for i in range(n)]
    j0 = min(range(n), key=lambda j: G[j][j] - 2 * h[j])  # nearest vertex
    x = [0.0] * n
    x[j0] = 1.0
    P = [j0]
    for _ in range(max_iter):
        g = [sum(G[i][j] * x[j] for j in P) - h[i] for i in range(n)]
        nu = -sum(g[j] for j in P) / len(P)
        cand = [j for j in range(n) if j not in P and g[j] + nu < -1e-12 * (1 + abs(nu))]
        if not cand:
            break
        P.append(min(cand, key=lambda j: g[j] + nu))
        while True:
            k = len(P)
            K = [[G[i][j] for j in P] + [1.0] for i in P] + [[1.0] * k + [0.0]]
            sol = _solve(K, [h[i] for i in P] + [1.0])
            if sol is None:
                P.pop()
                return x
            z = sol[:k]
            if min(z) > 0:
                for j in range(n):
                    x[j] = 0.0
                for i, j in enumerate(P):
                    x[j] = z[i]
                break
            alpha = min(x[j] / (x[j] - z[i]) for i, j in enumerate(P) if z[i] <= 0)
            for i, j in enumerate(P):
                x[j] += alpha * (z[i] - x[j])
            P = [j for j in P if x[j] > 1e-15]
            for j in range(n):
                if j not in P:
                    x[j] = 0.0
    return x


def esl_archetypes(X, r, max_iter=200, tol=1e-10):
    r"""Archetypal analysis: :math:`X \approx WH`, :math:`H = BX` (ESL 14.75-14.77).

    Minimises :math:`J(W,B) = \|X - WBX\|^2` with the rows of W (N x r) and of
    B (r x N) on the unit simplex, so every data point is a convex combination
    of r archetypes (rows of H) that are themselves convex combinations of the
    data. The minimisation alternates exact convex steps: each row of W by
    simplex-constrained least squares on the archetypes, then each archetype
    in turn (block coordinate descent) -- with the others fixed the criterion
    is :math:`\|w_k\|^2 \|h_k - R_k^T w_k/\|w_k\|^2\|^2` + const, so
    :math:`b_k` is the simplex least-squares fit of that target by the data
    points. J therefore never increases; it converges to a local minimum.
    The start places the archetypes by farthest-point traversal from the
    centroid (deterministic).

    Parameters
    ----------
    X : N x p nested sequence
    r : int
        Number of archetypes, 1 <= r <= N.
    max_iter, tol
        Stop when J changes by less than ``tol`` relatively.

    Returns
    -------
    RichResult
        ``archetypes`` (H, r x p), ``W``, ``B``, ``rss`` (J), ``rss_path``,
        ``iterations``, ``converged``.

    References
    ----------
    Cutler, A. & Breiman, L. (1994). Archetypal analysis. Technometrics 36, 338-347.

    Examples
    --------
    >>> X = [[0, 0], [4, 0], [4, 3], [0, 3], [1, 1], [2, 2], [3, 1]]
    >>> sorted(tuple(round(v, 6) for v in h) for h in esl_archetypes(X, 4)["archetypes"])
    [(0.0, 0.0), (0.0, 3.0), (4.0, 0.0), (4.0, 3.0)]
    """
    X = [[float(v) for v in x] for x in X]
    N = len(X)
    if N == 0 or not 1 <= r <= N:
        raise ValueError("need 1 <= r <= N")
    p = len(X[0])
    Xt = [list(c) for c in zip(*X)]
    cols = X  # data points as columns of X^T
    mu = [sum(c) / N for c in Xt]

    def d2(a, b):
        return sum((u - v) ** 2 for u, v in zip(a, b))

    idx = [max(range(N), key=lambda i: d2(X[i], mu))]
    while len(idx) < r:
        idx.append(max((i for i in range(N) if i not in idx), key=lambda i: min(d2(X[i], X[j]) for j in idx)))
    B = [[1.0 if i == j else 0.0 for i in range(N)] for j in idx]
    H = [X[j][:] for j in idx]

    def crit(W):
        return sum(d2(X[i], [sum(W[i][k] * H[k][c] for k in range(r)) for c in range(p)]) for i in range(N))

    W = [_simplex_ls(H, x) for x in X]
    J = crit(W)
    path = [J]
    conv = False
    it = 0
    for _ in range(max_iter):
        it += 1
        for k in range(r):
            wk = [W[i][k] for i in range(N)]
            s = sum(v * v for v in wk)
            if s == 0:
                continue
            R = [[X[i][c] - sum(W[i][j] * H[j][c] for j in range(r) if j != k) for c in range(p)] for i in range(N)]
            target = [sum(wk[i] * R[i][c] for i in range(N)) / s for c in range(p)]
            B[k] = _simplex_ls(cols, target)
            H[k] = [sum(B[k][i] * X[i][c] for i in range(N)) for c in range(p)]
        W = [_simplex_ls(H, x) for x in X]
        new = crit(W)
        path.append(new)
        if abs(J - new) <= tol * max(J, 1e-300) or new < 1e-24:
            J = new
            conv = True
            break
        J = new
    return RichResult(
        title="Archetypal analysis",
        summary_lines=[("archetypes", r), ("rss", J), ("iterations", it)],
        payload={
            "archetypes": H,
            "W": W,
            "B": B,
            "rss": J,
            "rss_path": path,
            "iterations": it,
            "converged": conv,
        },
    )


def cheatsheet() -> str:
    return "esl_archetypes -> archetypal analysis: data as convex mixtures of r archetypes (ESL 14.75-14.77)."
