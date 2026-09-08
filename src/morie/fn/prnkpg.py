# morie.fn -- function file (rootcoder007/morie)
"""PageRank -- alias of :mod:`morie.fn.pgrank`.

This module used to carry its own placeholder.  PageRank is already
implemented three-way in ``pgrank`` (Python, both R trees), so this is a
name, not a second implementation: a duplicate would be one more place
for the dangling-node mass to be handled differently.
"""

from .pgrank import pagerank as _pagerank

__all__ = ["pagerank"]


def pagerank(G, damping=0.85, n_iter=100):
    """PageRank of ``G`` with damping ``damping``; see ``pgrank.pagerank``.

    Formula: ``x = (1 - d)/n + d A^T D^-1 x``.

    References
    ----------
    Page, L., Brin, S., Motwani, R. & Winograd, T. (1999).  The PageRank
    citation ranking: bringing order to the web.  Stanford InfoLab
    technical report 1999-66.
    """
    return _pagerank(G, d=damping, n_iter=n_iter)


def cheatsheet():
    return "prnkpg: PageRank (alias of pgrank)"
