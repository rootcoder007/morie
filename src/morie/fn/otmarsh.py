# morie.fn -- function file (rootcoder007/morie)
"""Optimal transport after a marginal shift removes some source mass."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_marginal_shift"]


def ot_marginal_shift(a, b, C, delta):
    """Solve transport once part of the source marginal has been deleted.

    Deleting mass is the discrete face of Caffarelli and McCann's obstacle
    problem: the surviving support is an unknown free boundary, so the
    shifted problem is not the original plan with rows thinned out -- the
    optimum reorganises.  ``delta`` is taken as a total to remove, spread
    over the bins in proportion to ``a``, or as a per-bin vector when a
    vector is supplied.

    Formula: solve ``min_T <T,C>`` with ``a' = a - delta`` and total
    transported mass ``sum(a')``, i.e. the partial-transport problem of
    Caffarelli & McCann (2010).

    Parameters
    ----------
    a, b : array-like
        Source and target weights.
    C : array-like, shape (n, m)
        Ground cost.
    delta : float or array-like
        Mass removed from the source, in total or per bin.

    Returns
    -------
    RichResult
        ``T``, ``cost``, ``a_shift``, ``removed``, ``mass``, ``n``, ``m``.

    References
    ----------
    Caffarelli, L. A. and McCann, R. J. (2010).  Free boundaries in
    optimal transport and Monge-Ampere obstacle problems.  Annals of
    Mathematics 171(2):673-730.  doi:10.4007/annals.2010.171.673.
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cm = core.mat(C)
    n, m = len(aa), len(bb)
    if len(Cm) != n or len(Cm[0]) != m:
        raise ValueError("cost matrix does not match the marginals")
    dv = core.vec(delta)
    if len(dv) == 1:
        tot = sum(aa)
        if tot <= 0.0:
            raise ValueError("the source marginal has no mass to remove")
        d = [float(dv[0]) * aa[i] / tot for i in range(n)]
    elif len(dv) == n:
        d = [float(t) for t in dv]
    else:
        raise ValueError("delta must be a scalar or one value per source bin")
    ash = [aa[i] - d[i] for i in range(n)]
    if any(t < -1e-12 for t in ash):
        raise ValueError("delta removes more mass than a bin holds")
    ash = [t if t > 0.0 else 0.0 for t in ash]
    P, cost = ot.partial_plan(ash, bb, Cm, sum(ash))
    return RichResult(payload={
        "T": P, "cost": cost, "a_shift": ash, "removed": sum(d),
        "mass": sum(P[i][j] for i in range(n) for j in range(m)),
        "n": n, "m": m,
        "method": "Optimal transport under a marginal shift"})


def cheatsheet():
    return "otmarsh: transport after removing mass from the source marginal"
