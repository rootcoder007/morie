# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the natural total effect."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_natural_total"]


def tmle_natural_total(y, D, M, X):
    """Total effect by TMLE, with the mediator held out of the model.

    The natural total effect is the direct and indirect effects added
    back together, so the mediator must NOT enter the outcome model --
    conditioning on it would block the very path being counted.  The
    mediator is therefore used only for the decomposition report, never
    for adjustment.

    Formula: ``NTE = E[Y(1) - Y(0)] = NDE + NIE``, targeted with the
    clever covariate ``H = D / g - (1 - D) / (1 - g)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    M : array-like, shape (n,)
        Mediator; reported on, not adjusted for.
    X : array-like, shape (n, p)
        Baseline covariates.

    Returns
    -------
    RichResult
        ``estimate`` (NTE), ``se``, ``eps``, ``nde_naive`` (the same
        target with the mediator wrongly adjusted for, shown for
        contrast), ``n``.

    References
    ----------
    VanderWeele, T. J. (2015).  Explanation in Causal Inference.
    Oxford University Press, chapter 2.  The targeting step is van der
    Laan, M. J. & Rubin, D. (2006), International Journal of
    Biostatistics 2(1):11.
    """
    W = C.cbind1(C.mat(X))
    r = S.tmle(y, D, W)
    Wm = [list(W[i]) + [C.vec(M)[i]] for i in range(len(W))]
    r2 = S.tmle(y, D, Wm)
    return RichResult(payload={
        "estimate": r["psi"], "se": r["se"], "eps": r["eps"],
        "nde_naive": r2["psi"], "n": r["n"],
        "method": "TMLE for the natural total effect"})


def cheatsheet():
    return "tmlnte: TMLE for the natural total effect."
