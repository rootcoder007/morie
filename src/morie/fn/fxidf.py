# morie.fn -- function file (rootcoder007/morie)
"""Effect modification on the additive scale."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["effect_modification"]


def effect_modification(Y, X, C_mod):
    """Test whether the effect of X differs across levels of a modifier.

    VanderWeele point is that effect modification and interaction are
    different claims.  Interaction asks what happens when you set both
    variables; effect modification only asks whether the effect of one
    varies across strata defined by the other, and the modifier need not
    be causal at all -- it can be a proxy, a marker, or a stratifying
    label.  Only the exposure needs to be unconfounded for the
    modification statement to hold.

    Formula: fit ``Y = g0 + g1 X + g2 V + g3 X V``; ``g3`` is the
    additive modification, and the stratum-specific effects are
    ``g1 + g3 v``.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Exposure.
    C_mod : array-like, shape (n,)
        Candidate effect modifier.

    Returns
    -------
    RichResult
        ``estimate`` (the interaction coefficient ``g3``), ``se``,
        ``t``, ``coef``, ``effect_at_0``, ``effect_at_1``, ``n``.

    References
    ----------
    VanderWeele, T. J. (2009).  On the distinction between interaction
    and effect modification.  Epidemiology 20:863-871.
    """
    y = C.vec(Y)
    x = C.vec(X)
    v = C.vec(C_mod)
    n = len(y)
    des = [[1.0, x[i], v[i], x[i] * v[i]] for i in range(n)]
    beta, fitted, resid, xtxinv = C.lstsq(des, y)
    dof = n - 4
    s2 = sum(t * t for t in resid) / dof if dof > 0 else float("nan")
    se = math.sqrt(s2 * xtxinv[3][3]) if dof > 0 and xtxinv[3][3] > 0 else float("nan")
    return RichResult(payload={
        "estimate": beta[3], "se": se,
        "t": beta[3] / se if se == se and se > 0 else float("nan"),
        "coef": beta, "effect_at_0": beta[1], "effect_at_1": beta[1] + beta[3],
        "n": n, "method": "Additive effect modification, X by V interaction"})


def cheatsheet():
    return "fxidf: Effect modification on the additive scale."
