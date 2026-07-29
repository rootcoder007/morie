# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greedy navigable-small-world descent for approximate nearest
neighbour search (Malkov and Yashunin 2020; Alammar Ch 8)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_approximate_nearest_neighbor"]


def alammar_approximate_nearest_neighbor(query_vec, index, ef_search=8):
    """Greedy descent on a proximity graph, then beam refinement.

    ``index`` is ``{"points": (n, d) array, "neighbors": list of
    neighbour-index lists, "entry": start index}``. The search is
    APPROXIMATE by construction; the payload reports whether the
    result equals the true nearest neighbour (computed exactly here),
    because an ANN whose accuracy is never measured is just a slow
    hash.

    References: Alammar and Grootendorst, Ch 8; Malkov and Yashunin
    (2020).
    """
    q = np.atleast_1d(np.asarray(query_vec, dtype=float))
    P = np.atleast_2d(np.asarray(index["points"], dtype=float))
    nbrs = index["neighbors"]
    entry = int(index.get("entry", 0))
    ef = int(ef_search)
    if ef < 1:
        raise ValueError("ef_search must be positive.")
    if len(nbrs) != P.shape[0]:
        raise ValueError("need one neighbour list per point.")
    if not 0 <= entry < P.shape[0]:
        raise ValueError("entry point out of range.")
    if P.shape[1] != len(q):
        raise ValueError("query dimension does not match the index.")

    def d(i):
        return float(np.linalg.norm(P[i] - q))

    cur = entry
    hops = [cur]
    improved = True
    while improved:
        improved = False
        for j in nbrs[cur]:
            if d(int(j)) < d(cur):
                cur = int(j)
                hops.append(cur)
                improved = True
    # beam refinement around the greedy result
    cand = {cur}
    frontier = [cur]
    while frontier and len(cand) < ef:
        nxt = []
        for c in frontier:
            for j in nbrs[c]:
                j = int(j)
                if j not in cand:
                    cand.add(j)
                    nxt.append(j)
        frontier = nxt
    best = min(cand, key=d)
    true_best = int(np.argmin(np.linalg.norm(P - q, axis=1)))
    return RichResult(payload={
        "nearest": int(best), "distance": d(best),
        "greedy_path": hops, "candidates_examined": len(cand),
        "exact_nearest": true_best,
        "found_exact": best == true_best,
        "estimate": float(best), "n": P.shape[0],
        "method": "Greedy NSW descent + beam (Malkov and Yashunin 2020)"})


def cheatsheet():
    return "alann: greedy graph descent, beam refine, accuracy measured vs exact"
