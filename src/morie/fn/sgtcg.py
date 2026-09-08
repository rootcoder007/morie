# morie.fn -- function file (rootcoder007/morie)
"""Commute-time distance."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["commdist", "sgt_commute_distance"]


def commdist(A, tol=1e-09):
    """Commute-time distance.

    C_ij = 2m (L^+_ii + L^+_jj - 2 L^+_ij) = 2m R_ij.

    Expected commute time of the simple random walk, which is the
    resistance distance scaled by twice the total edge weight (Chandra
    et al. 1989; the resistance identity is Klein & Randic 1993).

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Commute-time distance", payload=_c.commdist(A=A, tol=tol))


sgt_commute_distance = commdist


def cheatsheet():
    return "sgtcg: Commute-time distance"
