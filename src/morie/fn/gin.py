# morie.fn -- function file (rootcoder007/morie)
"""Graph isomorphism network aggregation."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["ginagg", "gin"]


def ginagg(A, H, eps=0.0):
    """Graph isomorphism network aggregation.

    GIN aggregation: h_v <- (1 + eps) h_v + sum_{u in N(v)} h_u.

    Xu et al. (2019), Graph Isomorphism Network.  The (1 + eps) factor
    on the centre node is what keeps the self-representation
    distinguishable from the neighbour sum, so the aggregator is
    injective on multisets.  The learned MLP that follows is left to the
    caller; this is the aggregation step itself.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Graph isomorphism network aggregation", payload=_c.ginagg(A=A, H=H, eps=eps))


gin = ginagg


def cheatsheet():
    return "gin: Graph isomorphism network aggregation"
