# morie.fn -- function file (rootcoder007/morie)
"""TMLE for an interventional direct effect."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_nde_interventional"]


def tmle_nde_interventional(y, D, M, X, a_ref=0.0):
    """Direct effect with the mediator drawn from a fixed reference arm.

    Fixing the mediator distribution at the reference arm rather than at
    each person own counterfactual value is what makes this identified
    when an exposure-induced confounder sits between mediator and
    outcome.  The mediator model is fitted on the reference arm only,
    and its fitted values are then imposed on everyone.

    Formula: ``E[Y(1, M_{a*}) - Y(0, M_{a*})]``, targeted with
    ``H = D / g - (1 - D) / (1 - g)`` after replacing each mediator by
    its reference-arm prediction.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    M : array-like, shape (n,)
        Mediator.
    X : array-like, shape (n, p)
        Baseline covariates.
    a_ref : float, default 0.0
        Arm whose mediator distribution is imposed.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``m_shift`` (mean displacement
        of the imposed mediator from the observed one), ``n``.

    References
    ----------
    Vansteelandt, S. & Daniel, R. M. (2017).  Interventional effects
    for mediation analysis with multiple mediators.  Epidemiology
    28:258-265.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    Mv = C.vec(M)
    n = len(yv)
    Xm = C.cbind1(C.mat(X))
    ref = [i for i in range(n) if abs(Dv[i] - a_ref) < 0.5]
    mb, _, _, _ = S.ols([Xm[i] for i in ref], [Mv[i] for i in ref])
    Mhat = [C.dot(Xm[i], mb) for i in range(n)]
    des = [[Dv[i], Mv[i]] + list(Xm[i]) for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)
    gb = S.glmbin(Xm, Dv)
    g = [S.clip(S.expit(C.dot(Xm[i], gb)), 0.025, 0.975) for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    Q = [C.dot([Dv[i], Mv[i]] + list(Xm[i]), qb) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Q[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1 = [C.dot([1.0, Mhat[i]] + list(Xm[i]), qb) + eps / g[i] for i in range(n)]
    Q0 = [C.dot([0.0, Mhat[i]] + list(Xm[i]), qb) - eps / (1.0 - g[i]) for i in range(n)]
    psi = sum(Q1[i] - Q0[i] for i in range(n)) / n
    Qs = [Q[i] + eps * H[i] for i in range(n)]
    ic = [H[i] * (yv[i] - Qs[i]) + Q1[i] - Q0[i] - psi for i in range(n)]
    mi = sum(ic) / n
    se = math.sqrt(sum((v - mi) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    shift = sum(Mhat[i] - Mv[i] for i in range(n)) / n
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps,
        "m_shift": shift, "n": n,
        "method": "TMLE for an interventional direct effect"})


def cheatsheet():
    return "tmlnde2: TMLE for an interventional direct effect."
