# morie.fn -- function file (rootcoder007/morie)
"""PageRank.

DUPLICATE: PageRank is already implemented in ``pgrank``.  Per
ledger/wave2/DUPMAP.tsv this module aliases it instead of carrying a
third power iteration (``prnkpg`` and ``sgtpgr`` are already aliases of
the same function).
"""

from .pgrank import pagerank as _pagerank

__all__ = ["pagerank"]


def pagerank(A, alpha=0.85, n_iter=100):
    """Stationary distribution of the damped random surfer on ``A``.

    Alias of :func:`morie.fn.pgrank.pagerank`.  ``alpha`` is the damping
    factor, called ``d`` there.

    Formula: ``x = (1 - alpha)/n + alpha A^T D^-1 x``, with the mass of
    dangling nodes spread uniformly so the vector sums to one.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency; ``A[i][j]`` non-zero means a link from i to j.
    alpha : float, default 0.85
        Damping factor.
    n_iter : int, default 100
        Power iterations.

    Returns
    -------
    RichResult
        ``pr``, ``estimate``, ``top``, ``n``.

    References
    ----------
    Page, L., Brin, S., Motwani, R. & Winograd, T. (1999).  The PageRank
    citation ranking: bringing order to the web.  Stanford InfoLab
    Technical Report 1999-66.
    """
    return _pagerank(A, d=alpha, n_iter=n_iter)


def cheatsheet():
    return "pagrk: PageRank (alias of pgrank.pagerank)"
