# morie.fn -- function file (rootcoder007/morie)
"""Maximum bipartite matching."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["bipartite_matching"]


def bipartite_matching(edges, n_left=None, n_right=None):
    r"""Maximum-cardinality matching by augmenting paths (Hopcroft-Karp style).

    Finds the largest set of edges no two of which share a vertex. The
    algorithm repeatedly searches for an augmenting path -- one alternating
    between unmatched and matched edges, starting and ending unmatched -- and
    flips it, which raises the matching size by one. Berge's theorem
    guarantees a matching is maximum exactly when no augmenting path exists,
    which is what makes the stopping rule correct rather than heuristic.

    In causal work this is the exact solution to matched-pair designs, where
    greedy nearest-neighbour matching is not: greedy takes the best available
    control for each treated unit in turn, and an early greedy choice can
    strand a later treated unit with no acceptable partner. Optimal matching
    considers the assignment as a whole.

    ``n_unmatched_left`` is the number the design could not pair. Reporting it
    matters for the same reason as in caliper matching -- unmatched units are
    dropped, and dropping them changes the estimand.

    Parameters
    ----------
    edges : array-like
        Pairs ``(i, j)`` of admissible left-right matches, 0-indexed.
    n_left, n_right : int, optional
        Vertex counts. Inferred from ``edges`` otherwise.

    Returns
    -------
    RichResult
        ``matching`` (left index to right index, -1 if unmatched),
        ``size``, ``n_unmatched_left``, ``is_perfect``.

    References
    ----------
    Hopcroft, J. E., & Karp, R. M. (1973). An n^5/2 algorithm for maximum
        matchings in bipartite graphs. *SIAM Journal on Computing*, 2(4),
        225-231.

    Examples
    --------
    A perfect matching is found when one exists.

    >>> r = bipartite_matching([(0, 0), (1, 1), (2, 2)])
    >>> int(r["size"]), bool(r["is_perfect"])
    (3, True)

    Optimal matching beats greedy where a greedy first choice would strand a
    later unit: here left 0 can take either right, but only left 1 can take
    right 1.

    >>> r2 = bipartite_matching([(0, 0), (0, 1), (1, 1)])
    >>> int(r2["size"])
    2

    Unmatched left vertices are counted, since dropping them changes any
    downstream estimand.

    >>> r3 = bipartite_matching([(0, 0), (1, 0), (2, 0)])
    >>> int(r3["size"]), int(r3["n_unmatched_left"])
    (1, 2)

    >>> bipartite_matching([])
    Traceback (most recent call last):
        ...
    ValueError: edges must be non-empty
    """
    E = np.atleast_2d(np.asarray(edges, dtype=int))
    if E.size == 0:
        raise ValueError("edges must be non-empty")
    if E.shape[1] != 2:
        raise ValueError("edges must be pairs (i, j)")
    nL = int(E[:, 0].max()) + 1 if n_left is None else int(n_left)
    nR = int(E[:, 1].max()) + 1 if n_right is None else int(n_right)
    adj = [[] for _ in range(nL)]
    for i, j in E:
        adj[int(i)].append(int(j))

    match_l = np.full(nL, -1, dtype=int)
    match_r = np.full(nR, -1, dtype=int)

    def augment(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            # Berge: flip an alternating path that starts and ends unmatched.
            if match_r[v] == -1 or augment(match_r[v], seen):
                match_l[u], match_r[v] = v, u
                return True
        return False

    size = 0
    for u in range(nL):
        if match_l[u] == -1 and augment(u, np.zeros(nR, dtype=bool)):
            size += 1
    unmatched = int(np.sum(match_l == -1))
    return RichResult(
        title="Maximum bipartite matching",
        summary_lines=[("left", nL), ("right", nR), ("matched", int(size)),
                       ("unmatched left", unmatched)],
        warnings=(["unmatched units are dropped downstream, which changes the "
                   "estimand"] if unmatched else []),
        payload={
            "matching": match_l, "matching_right": match_r,
            "size": int(size), "n_unmatched_left": unmatched,
            "is_perfect": bool(size == min(nL, nR)),
            "n_left": nL, "n_right": nR, "method": "bipartite_matching",
        },
    )


def cheatsheet():
    return "bipMch: augmenting paths, maximum by Berge's theorem; greedy matching can strand later units"
