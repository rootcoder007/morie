# morie.fn -- function file (rootcoder007/morie)
"""Kirchhoff index."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["kirchidx", "sgt_kirchhoff_index"]


def kirchidx(A, tol=1e-09):
    """Kirchhoff index.

    Kf = (1/2) sum_ij R_ij = n sum_{k>0} 1/lambda_k   (Klein & Randic 1993).

    The Kirchhoff index.  Both forms are returned because their
    agreement is the paper's identity and a useful check on the
    spectral computation.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Kirchhoff index", payload=_c.kirchidx(A=A, tol=tol))


sgt_kirchhoff_index = kirchidx


def cheatsheet():
    return "sgtkir: Kirchhoff index"
