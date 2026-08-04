# morie.fn -- function file (rootcoder007/morie)
"""Resistance distance matrix."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["resdist", "sgt_resistance_distance_matrix"]


def resdist(A, tol=1e-09):
    """Resistance distance matrix.

    R_ij = L^+_ii + L^+_jj - 2 L^+_ij   (Klein & Randic 1993).

    Effective resistance between every pair of nodes when each edge is a
    unit conductance.  Unlike the shortest-path distance it falls when a
    parallel route is added, which is the property the paper is about.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Resistance distance matrix", payload=_c.resdist(A=A, tol=tol))


sgt_resistance_distance_matrix = resdist


def cheatsheet():
    return "sgtreg: Resistance distance matrix"
