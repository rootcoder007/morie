# morie.fn -- function file (rootcoder007/morie)
"""PageRank by power iteration -- alias of :mod:`morie.fn.pgrank`.

``pgrank`` already runs a fixed number of power iterations in all three
arms.  ``tol`` is accepted for signature compatibility and ignored: an
early stop on a tolerance is exactly the kind of thing that makes two
arms disagree in the last digits, and the fixed iteration count is the
deliberate choice made there.
"""

from .pgrank import pagerank as _pagerank

__all__ = ["sgt_pagerank_power"]


def sgt_pagerank_power(A, d=0.85, max_iter=100, tol=None):
    """PageRank vector of ``A``; see ``pgrank.pagerank``.

    Formula: ``p = (1 - d)/n + d M^T p``, iterated ``max_iter`` times.

    References
    ----------
    Page, L., Brin, S., Motwani, R. & Winograd, T. (1999).  The PageRank
    citation ranking: bringing order to the web.  Stanford InfoLab
    technical report 1999-66.
    """
    return _pagerank(A, d=d, n_iter=max_iter)


def cheatsheet():
    return "sgtpgr: PageRank via power iteration (alias of pgrank)"
