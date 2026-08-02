# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoverScore: Word Mover's Distance over contextual embeddings."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_moverscore", "word_movers_distance"]


def _cost_matrix(H, R, metric):
    if metric == "euclidean":
        diff = H[:, None, :] - R[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    nh = np.linalg.norm(H, axis=1)
    nr = np.linalg.norm(R, axis=1)
    if np.any(nh == 0) or np.any(nr == 0):
        raise ValueError(
            "a zero embedding has no direction; cosine distance is "
            "undefined.")
    return 1.0 - (H / nh[:, None]) @ (R / nr[:, None]).T


def word_movers_distance(cost, p, q, max_iter=10000):
    """Exact solution of the transportation problem min sum T_ij c_ij
    subject to row sums p and column sums q.

    Successive shortest augmenting paths with SPFA on the residual
    graph -- an exact min-cost-flow, not the relaxed lower bound that
    is often passed off as WMD.
    """
    n, m = cost.shape
    S, T = n + m, n + m + 1
    V = n + m + 2
    to, cap, cst, nxt = [], [], [], []
    head = [-1] * V

    def add(u, v, c, w):
        for (a, b, cc, ww) in ((u, v, c, w), (v, u, 0.0, -w)):
            to.append(b); cap.append(cc); cst.append(ww)
            nxt.append(head[a]); head[a] = len(to) - 1

    for i in range(n):
        add(S, i, float(p[i]), 0.0)
    for j in range(m):
        add(n + j, T, float(q[j]), 0.0)
    for i in range(n):
        for j in range(m):
            add(i, n + j, float("inf"), float(cost[i, j]))

    total_cost = 0.0
    total_flow = 0.0
    target = float(min(np.sum(p), np.sum(q)))
    it = 0
    while total_flow < target - 1e-12:
        it += 1
        if it > max_iter:
            raise ValueError(
                "the transportation problem did not converge within "
                f"{max_iter} augmentations.")
        dist = [np.inf] * V
        inq = [False] * V
        pre = [-1] * V
        dist[S] = 0.0
        queue = [S]
        inq[S] = True
        while queue:
            u = queue.pop(0)
            inq[u] = False
            e = head[u]
            while e != -1:
                if cap[e] > 1e-15 and dist[u] + cst[e] < dist[to[e]] - 1e-15:
                    dist[to[e]] = dist[u] + cst[e]
                    pre[to[e]] = e
                    if not inq[to[e]]:
                        inq[to[e]] = True
                        queue.append(to[e])
                e = nxt[e]
        if not np.isfinite(dist[T]):
            raise ValueError(
                "no augmenting path remains; supplies and demands are "
                "inconsistent.")
        push = np.inf
        v = T
        while v != S:
            e = pre[v]
            push = min(push, cap[e])
            v = to[e ^ 1]
        v = T
        while v != S:
            e = pre[v]
            cap[e] -= push
            cap[e ^ 1] += push
            v = to[e ^ 1]
        total_flow += push
        total_cost += push * dist[T]
    return total_cost


def kamath_moverscore(hypothesis_embeddings, reference_embeddings,
                      weights_h=None, weights_r=None, metric="euclidean",
                      normalizer=None):
    """MoverScore = 1 - WMD(emb(h), emb(r)) / normalizer.

    The transport is solved exactly (min-cost flow), so the distance is
    the real Word Mover's Distance and not a relaxation. Token weights
    default to uniform; pass IDF weights to reproduce the paper's
    variant. ``normalizer`` defaults to the largest single-pair cost,
    which bounds WMD from above and puts the score in [0, 1]; the
    value used is always reported, because "MoverScore = 0.8" is
    meaningless without it.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, MoverScore.

    Examples
    --------
    >>> H = [[0.0, 0.0], [1.0, 0.0]]
    >>> out = kamath_moverscore(H, H)
    >>> out["wmd"]
    0.0
    >>> out["estimate"]
    1.0
    >>> far = kamath_moverscore([[0.0]], [[2.0]])
    >>> far["wmd"], far["estimate"]
    (2.0, 0.0)
    >>> half = kamath_moverscore([[0.0], [2.0]], [[0.0], [0.0]])
    >>> half["wmd"]
    1.0
    """
    H = np.atleast_2d(np.asarray(hypothesis_embeddings, dtype=float))
    R = np.atleast_2d(np.asarray(reference_embeddings, dtype=float))
    if H.size == 0 or R.size == 0:
        raise ValueError("both token sets must be non-empty.")
    if H.shape[1] != R.shape[1]:
        raise ValueError(
            f"hypothesis embeddings are {H.shape[1]}-dim and reference "
            f"ones {R.shape[1]}-dim.")
    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric must be 'euclidean' or 'cosine'.")

    def _w(w, k, name):
        if w is None:
            return np.full(k, 1.0 / k)
        v = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
        if v.size != k:
            raise ValueError(f"{name} must have {k} entries; got {v.size}.")
        if np.any(v < 0):
            raise ValueError(f"{name} must be non-negative.")
        s = v.sum()
        if s == 0:
            raise ValueError(f"{name} sums to 0; there is nothing to move.")
        return v / s

    p = _w(weights_h, H.shape[0], "weights_h")
    q = _w(weights_r, R.shape[0], "weights_r")
    C = _cost_matrix(H, R, metric)
    wmd = word_movers_distance(C, p, q)
    norm = float(C.max()) if normalizer is None else float(normalizer)
    if norm <= 0:
        raise ValueError(
            "the normaliser is 0: every token pair is at distance 0, so "
            "1 - WMD/0 is undefined.")
    score = 1.0 - wmd / norm
    return RichResult(payload={
        "estimate": score, "score": score, "wmd": float(wmd),
        "normalizer": norm, "metric": metric,
        "n_hypothesis": int(H.shape[0]), "n_reference": int(R.shape[0]),
        "n": int(H.shape[0]),
        "method": "MoverScore = 1 - exact WMD / normalizer"})


def cheatsheet():
    return "kmmsc: 1 - exact Word Mover's Distance / max pair cost"
