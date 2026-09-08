# morie.fn -- function file (rootcoder007/morie)
"""Graph Laplacian pseudoinverse."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lappinv", "sgt_laplacian_pseudoinverse"]


def lappinv(A, tol=1e-09):
    """Graph Laplacian pseudoinverse.

    Moore-Penrose pseudoinverse of the graph Laplacian.

    L = D - A; L is singular because the all-ones vector spans its
    kernel on a connected graph, so L^+ is formed from the spectral
    decomposition with the zero modes dropped.  L^+ is basis
    independent, so it does not depend on the eigenvector convention.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Graph Laplacian pseudoinverse", payload=_c.lappinv(A=A, tol=tol))


sgt_laplacian_pseudoinverse = lappinv


def cheatsheet():
    return "sgtlpi: Graph Laplacian pseudoinverse"
