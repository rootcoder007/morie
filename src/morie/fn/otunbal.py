# morie.fn -- function file (rootcoder007/morie)
"""Unbalanced optimal transport by generalised Sinkhorn scaling."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_unbalanced"]


def ot_unbalanced(a, b, C, epsilon, lam, max_iter=200):
    """Transport between histograms of unequal mass.

    Hard marginal constraints force every unit of mass to be matched,
    which is exactly wrong when one measure is a noisy or truncated view
    of the other -- an outlier then drags a long-distance flow.  Replacing
    the constraints with KL penalties lets mass be created and destroyed
    at price ``lam``; ``lam -> infinity`` recovers the balanced problem.

    Formula: ``min_T <T,C> + eps H(T) + lam KL(T 1 | a) + lam KL(T' 1 |
    b)``, solved by the scalings ``u <- (a/(K v))^(lam/(lam+eps))``,
    ``v <- (b/(K' u))^(lam/(lam+eps))`` -- Peyre & Cuturi (2019)
    eq. (10.8)-(10.9), p. 163, rendered from the PDF; Chizat et al.
    (2018).

    Parameters
    ----------
    a, b : array-like
        Marginals; the totals need not agree.
    C : array-like, shape (n, m)
        Ground cost.
    epsilon : float
        Entropic strength, positive.
    lam : float
        Marginal-relaxation strength, positive.
    max_iter : int, default 200
        Scaling sweeps.

    Returns
    -------
    RichResult
        ``T``, ``cost``, ``mass``, ``mass_a``, ``mass_b``, ``n``, ``m``.

    References
    ----------
    Chizat, L., Peyre, G., Schmitzer, B. and Vialard, F.-X. (2018).
    Scaling algorithms for unbalanced optimal transport problems.
    Mathematics of Computation 87(314):2563-2609.  doi:10.1090/mcom/3303.
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cm = core.mat(C)
    n, m = len(aa), len(bb)
    if len(Cm) != n or len(Cm[0]) != m:
        raise ValueError("cost matrix does not match the marginals")
    T = ot.sinkhorn_unbalanced(aa, bb, Cm, float(epsilon), float(lam), max_iter)
    return RichResult(payload={
        "T": T, "cost": ot.frob(T, Cm),
        "mass": sum(T[i][j] for i in range(n) for j in range(m)),
        "mass_a": sum(aa), "mass_b": sum(bb), "n": n, "m": m,
        "method": "Unbalanced optimal transport"})


def cheatsheet():
    return "otunbal: unbalanced optimal transport with KL marginal penalties"


# compact alias per ledger/NAMING.md
otunbalanced = ot_unbalanced
