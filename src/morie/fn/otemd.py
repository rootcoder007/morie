# morie.fn -- function file (rootcoder007/morie)
"""Exact discrete optimal transport (the earth mover's distance)."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_emd_solver"]


def ot_emd_solver(a, b, C):
    """Exact earth mover's distance between two histograms.

    The linear program is solved outright, not smoothed: the answer is
    the true optimum of the Kantorovich problem rather than an entropic
    relaxation of it.  The plan returned is a vertex of the transport
    polytope, so at most ``n + m - 1`` of its entries are non-zero.

    Formula: ``min_T <T, C>`` subject to ``T 1 = a``, ``T' 1 = b``,
    ``T >= 0`` -- Kantorovich's problem, eq. (2.11) of Peyre & Cuturi
    (2019).  Solved by the transportation simplex: north-west-corner
    start, potentials from the basis tree, MODI pivoting.

    Parameters
    ----------
    a : array-like, shape (n,)
        Source weights.  Non-negative.
    b : array-like, shape (m,)
        Target weights.  Non-negative, same total mass as ``a``.
    C : array-like, shape (n, m)
        Ground cost.

    Returns
    -------
    RichResult
        ``T`` (the optimal plan), ``cost``, ``n``, ``m``, ``n_basic``.

    References
    ----------
    Rubner, Y., Tomasi, C. and Guibas, L. J. (2000).  The earth mover's
    distance as a metric for image retrieval.  International Journal of
    Computer Vision 40(2):99-121.  doi:10.1023/A:1026543900054.
    Peyre & Cuturi (2019), Computational Optimal Transport, eq. (2.11).
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cm = core.mat(C)
    T, cost = ot.emd(aa, bb, Cm)
    nb = sum(1 for row in T for t in row if t > 1e-15)
    return RichResult(payload={
        "T": T, "cost": cost, "n": len(aa), "m": len(bb), "n_basic": nb,
        "method": "Exact optimal transport (transportation simplex)"})


def cheatsheet():
    return "otemd: exact earth mover's distance between two histograms"


# compact alias per ledger/NAMING.md
otemdsolver = ot_emd_solver
