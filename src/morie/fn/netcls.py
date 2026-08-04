# morie.fn -- function file (rootcoder007/morie)
"""Closeness centrality of a node (re-export)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['netclocent', 'closeness_centrality']


def netclocent(A):
    """Closeness centrality of a node (re-export).

    Second listing of the same measure; delegates.


    Formula: see clocent

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix.

    Returns
    -------
    RichResult
        the payload of :func:`morie.fn.clocen.clocent`.

    References
    ----------
    Sabidussi (1966) for the sum-distance form and Freeman (1979),
    Centrality in social networks: conceptual clarification, Social
    Networks 1:215-239, for the (n-1)-normalised measure.  Freeman's
    article is paywalled; the normalisation C(v) = (n-1)/sum_u d(v,u)
    is as restated in the centrality literature that cites him.
    """
    from .clocen import clocent as _c
    return _c(A)


closeness_centrality = netclocent


def cheatsheet():
    return "netcls: Closeness centrality of a node (re-export)."
