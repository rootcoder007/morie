"""Automatic zoning procedure (AZP) for the zone-design, aggregation and redistricting problems.

Openshaw, S. (1977). A geographical solution to scale and aggregation problems in
region-building, partitioning and spatial modelling. Transactions of the Institute of British
Geographers 2, 459-472. Openshaw, S. and Rao, L. (1995). Algorithms for reengineering 1991
census geography. Environment and Planning A 27, 425-446.
"""

from ._metaheur import Rand
from ._regcore import canonical, local_search, neighbours, objective_fn
from ._richresult import RichResult

__all__ = ["automatic_zoning"]


def automatic_zoning(X, adjacency, k, objective="ssd", weights=None, init=None, seed=0, max_iter=1000):
    r"""Aggregate n contiguous units into k contiguous zones by AZP local search.

    The start is ``init`` or a random contiguous partition: k Philox-drawn seed units
    grow in turn, each claiming one unassigned neighbouring unit (drawn at random) per
    round until every unit is assigned. The AZP search then repeatedly moves border units
    into adjacent zones when the donor zone stays contiguous and non-empty and the
    objective strictly falls (Openshaw 1977, Openshaw and Rao 1995). ``objective="ssd"``
    minimises the within-zone sum of squared deviations of X; ``"balance"`` minimises
    sum_z (W_z - mean W)^2 for the summed ``weights`` (population), the redistricting form.

    Parameters
    ----------
    X : list of attribute vectors
    adjacency : 0/1 matrix or neighbour lists
    k : int
    objective : {"ssd", "balance"}
    weights : sequence, optional
        Unit weights for ``"balance"``.
    init : sequence of int, optional
        Initial contiguous zone labels.
    seed : int
    max_iter : int
        Maximum local-search sweeps.

    Returns
    -------
    RichResult
        Keys: labels, objective, initial, sweeps.

    References
    ----------
    Openshaw, S. (1977). Transactions of the Institute of British Geographers 2, 459-472.
    Openshaw, S. and Rao, L. (1995). Environment and Planning A 27, 425-446.

    Examples
    --------
    >>> X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]
    >>> automatic_zoning(X, [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]], 2)["labels"]
    [0, 0, 0, 1, 1, 1]
    """
    X = [[float(v) for v in r] for r in X]
    n = len(X)
    nb = neighbours(adjacency, n)
    nbset = [set(r) for r in nb]
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= number of units")
    if objective == "balance" and weights is None:
        raise ValueError('objective "balance" needs weights')
    if objective not in ("ssd", "balance"):
        raise ValueError('objective must be "ssd" or "balance"')
    rnd = Rand(seed)
    if init is None:
        lab = [-1] * n
        pool = list(range(n))
        seeds = []
        for z in range(k):
            s = pool.pop(rnd.idx(len(pool)))
            seeds.append(s)
            lab[s] = z
        while -1 in lab:
            grew = False
            for z in range(k):
                front = sorted({v for i in range(n) if lab[i] == z for v in nb[i] if lab[v] == -1})
                if front:
                    lab[front[rnd.idx(len(front))]] = z
                    grew = True
            if not grew:
                raise ValueError("the contiguity graph is not connected")
    else:
        lab = [int(v) for v in init]
    start = canonical(lab)
    obj = objective_fn(X, weights, objective)
    labels, val, sweeps = local_search(start, nbset, obj, rnd, lambda region: len(region) > 0, max_iter)
    return RichResult(
        title="Automatic zoning procedure",
        summary_lines=[("zones", k), ("objective", val)],
        payload={"labels": labels, "objective": val, "initial": start, "sweeps": sweeps},
    )


def cheatsheet():
    return "azpreg: AZP zone design / aggregation / redistricting (Openshaw 1977)"
