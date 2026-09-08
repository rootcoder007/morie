# morie.fn -- function file (rootcoder007/morie)
"""Eigenvector centrality from the leading eigenvector (re-export)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['sgteigcent', 'sgt_eigenvector_centrality']


def sgteigcent(A):
    """Eigenvector centrality from the leading eigenvector (re-export).

    Third listing of the same measure; delegates.


    Formula: see eigcent

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative adjacency matrix.

    Returns
    -------
    RichResult
        the payload of :func:`morie.fn.eigcen.eigcent`.

    References
    ----------
    Bonacich (1972), Factoring and weighting approaches to status scores
    and clique identification, Journal of Mathematical Sociology
    2:113-120.  Paywalled; the measure is the principal eigenvector of
    the adjacency matrix, as it is universally described in the
    centrality literature (e.g. Bonacich 2000, Social Networks
    22:357-365, which restates his own definition).
    """
    from .eigcen import eigcent as _e
    return _e(A)


sgt_eigenvector_centrality = sgteigcent


def cheatsheet():
    return "sgteig: Eigenvector centrality from the leading eigenvector (re-export)."
