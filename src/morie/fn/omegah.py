# morie.fn -- function file (rootcoder007/morie)
"""McDonald omega hierarchical."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["omega_hierarchical"]


def omega_hierarchical(X, loadings_g, loadings_specific=None):
    """Share of total score variance due to the single general factor.

    Cronbach alpha is routinely read as evidence that a scale measures
    one thing, and it is not: alpha is high whenever the items
    intercorrelate at all, whatever the factor structure.  Omega
    hierarchical answers the question alpha is mistaken for, by putting
    only the general-factor loadings in the numerator and the whole test
    variance in the denominator.

    Formula: ``omega_h = (sum_i lambda_gi)^2 / Var(T)`` with
    ``Var(T) = (sum lambda_g)^2 + sum_s (sum_i lambda_si)^2
    + sum_i psi_i`` and ``psi_i = 1 - lambda_gi^2 - lambda_si^2``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Item scores; used only for the observed total variance, which is
        reported alongside.
    loadings_g : array-like, shape (p,)
        General-factor loadings.
    loadings_specific : array-like, optional
        Group-factor loadings, shape (p,) with a group label implied by
        a second column, or (p,) for a single group factor.

    Returns
    -------
    RichResult
        ``estimate`` (omega_h), ``omega_total``, ``var_total``,
        ``uniqueness``, ``p``.

    References
    ----------
    Zinbarg, R. E., Revelle, W., Yovel, I. & Li, W. (2005).  Cronbach
    alpha, Revelle beta and McDonald omega_h: their relations with each
    other and two alternative conceptualizations of reliability.
    Psychometrika 70:123-133.  The coefficient is McDonald, R. P.
    (1999), Test Theory: A Unified Treatment, Erlbaum, chapter 6.
    """
    lg = C.vec(loadings_g)
    p = len(lg)
    ls = C.vec(loadings_specific) if loadings_specific is not None else [0.0] * p
    psi = [1.0 - lg[i] ** 2 - ls[i] ** 2 for i in range(p)]
    sg2 = sum(lg) ** 2
    ss2 = sum(ls) ** 2
    var_t = sg2 + ss2 + sum(psi)
    return RichResult(payload={
        "estimate": sg2 / var_t if var_t != 0.0 else float("nan"),
        "omega_total": (sg2 + ss2) / var_t if var_t != 0.0 else float("nan"),
        "var_total": var_t, "uniqueness": psi, "p": p,
        "method": "McDonald omega hierarchical"})


def cheatsheet():
    return "omegah: McDonald omega hierarchical."
