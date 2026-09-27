"""Flow-capturing location model (Hodgson 1990).

Hodgson, M. J. (1990). A flow-capturing location-allocation model. Geographical Analysis 22,
270-279.
"""

from ._facloc import search
from ._richresult import RichResult

__all__ = ["flow_capturing_location"]


def flow_capturing_location(p, paths, volumes=None, n_nodes=None, max_enum=200000):
    r"""Choose p nodes maximising the volume of origin-destination flows passing a chosen node.

    A flow is captured when its path contains at least one open node (Hodgson 1990,
    eqs. 1-4). Exact by enumeration when C(m, p) <= max_enum, otherwise greedy plus
    substitution.

    Parameters
    ----------
    p : int
    paths : list of node-index lists
        The node sequence of each flow's route.
    volumes : sequence, optional
        Flow volumes (default 1).
    n_nodes : int, optional
        Number of candidate nodes (default max node index + 1).
    max_enum : int

    Returns
    -------
    RichResult
        Keys: sites, captured (bool per flow), captured_volume, share, method.

    References
    ----------
    Hodgson, M. J. (1990). Geographical Analysis 22, 270-279.

    Examples
    --------
    >>> flow_capturing_location(1, [[0, 1, 2], [3, 1, 4], [5, 6]])["captured_volume"]
    2.0
    """
    P = [set(int(v) for v in path) for path in paths]
    q = [1.0] * len(P) if volumes is None else [float(v) for v in volumes]
    m = (max(max(s) for s in P if s) + 1) if n_nodes is None else int(n_nodes)

    def obj(S):
        s = 0.0
        for path, v in zip(P, q):
            if any(j in path for j in S):
                s += v
        return -s

    S, best, how = search(obj, m, int(p), max_enum)
    cap = [any(j in path for j in S) for path in P]
    tot = 0.0
    for v in q:
        tot += v
    return RichResult(
        title="Flow-capturing location",
        summary_lines=[("captured", -best), ("method", how)],
        payload={"sites": S, "captured": cap, "captured_volume": -best, "share": -best / tot, "method": how},
    )


def cheatsheet():
    return "fclmop: flow-capturing location model (Hodgson 1990)"
