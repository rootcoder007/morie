# morie.fn -- function file (rootcoder007/morie)
"""Partial optimal transport: move a prescribed amount of mass."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_partial_ot"]


def ot_partial_ot(a, b, C, m):
    """Transport only ``m`` units of mass and leave the rest in place.

    The Kantorovich problem insists that everything move.  Partial
    transport relaxes that to an inequality and asks only that a fixed
    budget of mass be matched, which is the right formulation when the two
    measures overlap on a sub-region -- Caffarelli and McCann's free
    boundary is precisely the edge of the matched part.  The problem is
    solved exactly by padding the cost with a zero-price dummy row and
    column, so the answer is a genuine LP optimum, not a relaxation.

    Formula: ``min_T <T,C>`` subject to ``T 1 <= a``, ``T' 1 <= b``,
    ``sum T = m``.

    Parameters
    ----------
    a, b : array-like
        Source and target weights.
    C : array-like, shape (n, k)
        Ground cost.
    m : float
        Mass to transport; must not exceed either total.

    Returns
    -------
    RichResult
        ``T``, ``cost``, ``mass``, ``a_left``, ``b_left``, ``n``, ``m_bins``.

    References
    ----------
    Caffarelli, L. A. and McCann, R. J. (2010).  Free boundaries in
    optimal transport and Monge-Ampere obstacle problems.  Annals of
    Mathematics 171(2):673-730.  doi:10.4007/annals.2010.171.673.
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cm = core.mat(C)
    if len(Cm) != len(aa) or len(Cm[0]) != len(bb):
        raise ValueError("cost matrix does not match the marginals")
    P, cost = ot.partial_plan(aa, bb, Cm, m)
    left_a = [aa[i] - sum(P[i]) for i in range(len(aa))]
    left_b = [bb[j] - sum(P[i][j] for i in range(len(aa)))
              for j in range(len(bb))]
    return RichResult(payload={
        "T": P, "cost": cost,
        "mass": sum(P[i][j] for i in range(len(aa)) for j in range(len(bb))),
        "a_left": left_a, "b_left": left_b,
        "n": len(aa), "m_bins": len(bb),
        "method": "Partial optimal transport"})


def cheatsheet():
    return "otpr: partial optimal transport of a prescribed mass"


# compact alias per ledger/NAMING.md
otpartialot = ot_partial_ot
