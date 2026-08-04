# morie.fn -- function file (rootcoder007/morie)
"""Kantorovich dual objective value."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ot_kantorovich_dual_value"]


def ot_kantorovich_dual_value(a, b, f, g):
    """Value of the dual objective for a candidate pair of potentials.

    The dual turns a search over couplings -- a huge object -- into a
    search over two functions on the marginals.  Any feasible pair gives
    a LOWER bound on the transport cost, so the value returned here is
    only the optimum when the potentials are optimal; feasibility
    against a cost matrix is not checked, because the caller usually has
    it and checking would cost more than the evaluation.

    Formula: ``<a, f> + <b, g>``.

    Parameters
    ----------
    a, b : array-like
        Source and target marginals.
    f, g : array-like
        Kantorovich potentials on the two marginals.

    Returns
    -------
    RichResult
        ``dual_val``, ``estimate``, ``n``, ``m``.

    References
    ----------
    Villani, C. (2003).  Topics in Optimal Transportation.  American
    Mathematical Society, Graduate Studies in Mathematics 58,
    theorem 1.3 (Kantorovich duality).
    """
    av, bv = C.vec(a), C.vec(b)
    fv, gv = C.vec(f), C.vec(g)
    val = sum(av[i] * fv[i] for i in range(len(av))) + sum(bv[j] * gv[j] for j in range(len(bv)))
    return RichResult(payload={"dual_val": val, "estimate": val, "n": len(av),
                               "m": len(bv), "method": "Kantorovich dual objective value"})


def cheatsheet():
    return "ototk: Kantorovich dual objective value."
