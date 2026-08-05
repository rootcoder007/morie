# morie.fn -- function file (rootcoder007/morie)
"""Leiden community detection -- an alias for :mod:`scleid`.

``ledger/wave2/DUPMAP.tsv`` records ``leid`` as a duplicate of
``scleid``, and it is: the same local-move / refinement / aggregation
optimisation of the same quality function.  Only the calling convention
differs, so that is all this module supplies.
"""

from .scleid import leiden_clustering

__all__ = ["leiden_communities"]


def leiden_communities(y, A, resolution=1.0, quality="modularity",
                       max_iter=20):
    """Partition a graph into well-connected communities.

    Louvain can leave a community internally disconnected -- a node moved
    away can sever the group it left behind, and nothing in the algorithm
    ever checks.  Leiden inserts a refinement phase that guarantees every
    returned community is connected, which is why its partitions survive
    being looked at.

    Formula: local move, refinement, then aggregation, repeated --
    Traag, Waltman & van Eck (2019).

    This is an alias.  The optimisation lives in ``morie.fn.scleid``.

    Parameters
    ----------
    y : ignored
        Accepted for interface compatibility with the stub signature.
    A : array-like, shape (n, n)
        Weighted adjacency matrix.
    resolution : float, default 1.0
        Resolution parameter of the quality function.
    quality : str, default 'modularity'
        Quality function, passed through.
    max_iter : int, default 20
        Passes.

    Returns
    -------
    RichResult
        Whatever ``scleid.leiden_clustering`` returns, unchanged.

    References
    ----------
    Traag, V. A., Waltman, L. and van Eck, N. J. (2019).  From Louvain to
    Leiden: guaranteeing well-connected communities.  Scientific Reports
    9:5233.  doi:10.1038/s41598-019-41695-z.
    """
    return leiden_clustering(A, resolution, quality, max_iter)


def cheatsheet():
    return "leid: Leiden community detection (alias of scleid)"
