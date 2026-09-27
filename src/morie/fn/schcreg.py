"""Spatially constrained hierarchical clustering (the REDCAP full-order agglomeration, Guo 2008).

Guo, D. (2008). Regionalization with dynamically constrained agglomerative clustering and
partitioning (REDCAP). International Journal of Geographical Information Science 22, 801-823.
"""

from ._regcore import neighbours
from ._richresult import RichResult
from .redcap import _LINKAGES, _agglomerate

__all__ = ["constrained_hierarchical"]


def constrained_hierarchical(X, adjacency, k, linkage="ward"):
    r"""Spatially constrained hierarchical clustering: the REDCAP full-order agglomeration stopped at k clusters.

    Parameters
    ----------
    X : list of attribute vectors
    adjacency : 0/1 matrix or neighbour lists
    k : int
    linkage : {"single", "complete", "average", "ward"}

    Returns
    -------
    RichResult
        Keys: labels (0-based, in order of first appearance), merges (tree edges in order).

    References
    ----------
    Guo, D. (2008). International Journal of Geographical Information Science 22, 801-823.

    Examples
    --------
    >>> X = [[0.0], [0.1], [5.0], [5.1]]
    >>> constrained_hierarchical(X, [[1], [0, 2], [1, 3], [2]], 2)["labels"]
    [0, 0, 1, 1]
    """
    if linkage not in _LINKAGES:
        raise ValueError("linkage must be single, complete, average or ward")
    X = [[float(v) for v in r] for r in X]
    n = len(X)
    if not 1 <= int(k) <= n:
        raise ValueError("need 1 <= k <= number of units")
    tree, labels = _agglomerate(X, neighbours(adjacency, n), linkage, "full", int(k))
    if labels is None:
        labels = list(range(n))
    return RichResult(
        title="Spatially constrained hierarchical clustering",
        summary_lines=[("clusters", int(k))],
        payload={"labels": labels, "merges": tree},
    )


def cheatsheet():
    return "schcreg: spatially constrained hierarchical clustering (Guo 2008)"
