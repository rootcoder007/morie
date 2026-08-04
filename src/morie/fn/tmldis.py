# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a disparity reduction."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_disparity"]


def tmle_disparity(y, S_grp, X, X_target=None):
    """How much of a group gap survives equalising the covariates.

    A disparity is not a causal effect of group membership -- race and
    sex are not things one assigns -- so the estimand is deliberately
    different: hold the group fixed and move the covariate distribution
    instead.  What is left after standardising to the reference
    distribution is the residual disparity, and the difference between
    the two is the part attributable to the covariates.

    Formula: ``PAD = E[Y|S = 1] - E_{X ~ P(X|S = 0)}[E(Y|S = 1, X)]``,
    estimated by standardising the group-1 outcome regression over the
    target covariate distribution.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    S_grp : array-like, shape (n,)
        Binary group indicator.
    X : array-like, shape (n, p)
        Covariates.
    X_target : array-like, optional
        Covariate rows defining the reference distribution; the
        ``S = 0`` rows by default.

    Returns
    -------
    RichResult
        ``estimate`` (the disparity remaining after standardisation),
        ``crude`` (unadjusted gap), ``explained``, ``se``, ``n``.

    References
    ----------
    VanderWeele, T. J. & Robinson, W. R. (2014).  On causal
    interpretation of race in regressions adjusting for confounding and
    mediating variables.  Epidemiology 25:473-484.
    """
    yv = C.vec(y)
    Sv = C.vec(S_grp)
    n = len(yv)
    W = C.cbind1(C.mat(X))
    i1 = [i for i in range(n) if Sv[i] > 0.5]
    i0 = [i for i in range(n) if Sv[i] <= 0.5]
    b1, _, _, _ = S.ols([W[i] for i in i1], [yv[i] for i in i1])
    Wt = C.cbind1(C.mat(X_target)) if X_target is not None else [W[i] for i in i0]
    std = sum(C.dot(row, b1) for row in Wt) / len(Wt)
    mu1 = sum(yv[i] for i in i1) / len(i1)
    mu0 = sum(yv[i] for i in i0) / len(i0)
    resid = [yv[i] - C.dot(W[i], b1) for i in i1]
    mr = sum(resid) / len(resid)
    se = math.sqrt(sum((v - mr) ** 2 for v in resid) / (len(resid) - 1) / len(resid))
    return RichResult(payload={
        "estimate": mu1 - std, "crude": mu1 - mu0,
        "explained": (mu1 - mu0) - (mu1 - std), "se": se, "n": n,
        "method": "Standardised disparity remaining after covariates"})


tmledisparity = tmle_disparity


def cheatsheet():
    return "tmldis: TMLE for a disparity reduction."
