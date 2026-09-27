"""Max-p regionalization (Duque, Anselin and Rey 2012).

Duque, J. C., Anselin, L. and Rey, S. J. (2012). The max-p-regions problem. Journal of Regional
Science 52, 397-419.
"""

from ._metaheur import Rand
from ._regcore import local_search, neighbours, objective_fn, ssd
from ._richresult import RichResult

__all__ = ["max_p_regions"]


def max_p_regions(X, adjacency, weights, threshold, n_construct=50, seed=0, local=True, max_iter=1000):
    r"""Partition contiguous units into the largest number p of contiguous regions whose summed
    ``weights`` reach ``threshold``, then minimise heterogeneity for that p.

    Construction (Duque et al. 2012, Sec. 4.1): units are visited in a Philox-shuffled
    order; each unassigned unit seeds a region that absorbs randomly chosen unassigned
    neighbours until the threshold is met, and a region that cannot reach it releases its
    units as enclaves; enclaves then join the adjacent region that raises the within-region
    sum of squared deviations least. The construction is repeated ``n_construct`` times,
    keeping the largest p and, among those, the smallest SSD; the AZP local search then
    lowers the SSD, moving only units whose donor region stays above the threshold.

    Parameters
    ----------
    X : list of attribute vectors
    adjacency : 0/1 matrix or neighbour lists
    weights : sequence
        Spatially extensive attribute (e.g. population).
    threshold : float
    n_construct : int
    seed : int
    local : bool
        Run the local search.
    max_iter : int

    Returns
    -------
    RichResult
        Keys: labels, p, ssd, region_weights.

    References
    ----------
    Duque, J. C., Anselin, L. and Rey, S. J. (2012). Journal of Regional Science 52, 397-419.

    Examples
    --------
    >>> X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]
    >>> max_p_regions(X, [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]], [1] * 6, 3)["p"]
    2
    """
    X = [[float(v) for v in r] for r in X]
    n = len(X)
    nb = neighbours(adjacency, n)
    nbset = [set(r) for r in nb]
    w = [float(v) for v in weights]
    tot = 0.0
    for v in w:
        tot += v
    if tot < threshold:
        raise ValueError("the total weight is below the threshold")
    rnd = Rand(seed)

    def wsum(nodes):
        s = 0.0
        for i in nodes:
            s += w[i]
        return s

    best = None
    for _ in range(int(n_construct)):
        lab = [-1] * n
        regions = []
        for i in sorted(range(n), key=lambda q: (rnd.u(), q)):
            if lab[i] != -1:
                continue
            reg = [i]
            lab[i] = len(regions)
            while wsum(reg) < threshold:
                front = sorted({v for u in reg for v in nb[u] if lab[v] == -1})
                if not front:
                    break
                v = front[rnd.idx(len(front))]
                lab[v] = len(regions)
                reg.append(v)
            if wsum(reg) >= threshold:
                regions.append(sorted(reg))
            else:
                for u in reg:
                    lab[u] = -2  # enclave
        if not regions:
            continue
        for u in range(n):
            if lab[u] == -2:
                lab[u] = -1
        while -1 in lab:
            moved = False
            for u in range(n):
                if lab[u] != -1:
                    continue
                cand = sorted({lab[v] for v in nb[u] if lab[v] >= 0})
                if cand:
                    r = min(cand, key=lambda q: (ssd(X, regions[q] + [u]) - ssd(X, regions[q]), q))
                    regions[r] = sorted(regions[r] + [u])
                    lab[u] = r
                    moved = True
            if not moved:
                raise ValueError("the contiguity graph is not connected")
        val = 0.0
        for r in regions:
            val += ssd(X, r)
        key = (-len(regions), val)
        if best is None or key < best[0]:
            best = (key, list(lab))
    if best is None:
        raise ValueError("no region reaches the threshold")
    labels = best[1]
    if local:
        labels, _, _ = local_search(
            labels, nbset, objective_fn(X, None, "ssd"), rnd, lambda region: wsum(region) >= threshold, max_iter
        )
    p = max(labels) + 1
    regions = [[i for i in range(n) if labels[i] == r] for r in range(p)]
    total = 0.0
    for r in regions:
        total += ssd(X, r)
    return RichResult(
        title="Max-p regions",
        summary_lines=[("p", p), ("SSD", total)],
        payload={"labels": labels, "p": p, "ssd": total, "region_weights": [wsum(r) for r in regions]},
    )


def cheatsheet():
    return "maxpreg: max-p regionalization (Duque, Anselin and Rey 2012)"
