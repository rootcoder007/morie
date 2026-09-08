"""Leiden refined community detection (alias to the scleid optimizer)."""

from ._richresult import RichResult  # noqa: F401  (re-export convention)
from .scleid import leiden_clustering

__all__ = ["lemR", "leiden_grph"]


def lemR(A, resolution=1.0, quality="modularity", max_iter=20):
    """
    Leiden community detection: local moving plus a refinement that
    guarantees connected communities.

    This is an alias: the optimisation lives in
    :func:`morie.fn.scleid.leiden_clustering` (deterministic index-order
    variant; every community is split into its connected components
    before aggregation, which is the guarantee the Leiden refinement
    exists to provide). Implementing a second copy here would only let
    the two drift apart.

    Sources
    -------
    Traag, V. A., Waltman, L. & van Eck, N. J. (2019). From Louvain to
    Leiden: guaranteeing well-connected communities. *Scientific
    Reports*, 9, 5233, arXiv:1810.08473, Sec. "Leiden algorithm" and
    eq. (2) (fetched-wave3/traag-2019-louvain-to-leiden.pdf).

    Parameters
    ----------
    A : array-like, (n, n)
        Weighted adjacency matrix.
    resolution : float
        Resolution parameter of the quality function.
    quality : str
        Quality function ("modularity" or "cpm"), passed through.
    max_iter : int
        Local-moving passes.

    Returns
    -------
    RichResult
        The scleid result unchanged: labels, estimate (quality),
        n_communities, connected, passes.
    """
    return leiden_clustering(A, resolution, quality, max_iter)


# long descriptive alias (stub-era name)
leiden_grph = lemR


def cheatsheet():
    return "lemR: Leiden refined community detection (alias of scleid)"

# public names resolved by fn/_lazy_map.json
leidengrph = lemR
