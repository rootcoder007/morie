# morie.fn -- tail3 batch (rootcoder007/morie)
"""Conditional indirect effect (moderated mediation).

Source consulted: Preacher, K.J., Rucker, D.D. & Hayes, A.F. (2007).
Addressing moderated mediation hypotheses: theory, methods, and
prescriptions.  *Multivariate Behavioral Research* 42(1), 185-227.  In their
Model 2, where the a path is moderated by W,

    M = a0 + a1 X + a2 W + a3 X W + ...
    Y = ... + b1 M + ...

the conditional indirect effect of X on Y through M is stated on p.197 as

    f(theta | W) = b1 (a1 + a3 W)

with the simple slope of the a path carrying the standard error of their
equation (8),

    SE(a1 + a3 W) = sqrt( s2_a1 + 2 s_a1a3 W + s2_a3 W^2 )

and the first-order multivariate delta method of their equation (13), applied
to this model, giving

    SE(f) = sqrt( (a1 + a3 W)^2 s2_b1 + b1^2 (s2_a1 + 2 s_a1a3 W + s2_a3 W^2) )

The normal-theory test divides the effect by that standard error, their
equation (14).  The paper prefers bootstrap confidence intervals; the
normal-theory quantities are what it gives in closed form, so they are what
is computed here.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["conditional_indirect_effect"]


def conditional_indirect_effect(a1, a3, b, w, sa1=None, sa3=None, sa1a3=0.0, sb=None):
    """Conditional indirect effect and its normal-theory standard error.

    Parameters
    ----------
    a1 : float
        Coefficient of X in the mediator model.
    a3 : float
        Coefficient of the X-by-W product in the mediator model.
    b : float
        Coefficient of M in the outcome model.
    w : float or array-like
        Value(s) of the moderator at which to condition.
    sa1, sa3, sb : float, optional
        Standard errors of ``a1``, ``a3`` and ``b``.
    sa1a3 : float
        Covariance of ``a1`` and ``a3``.

    Returns
    -------
    RichResult
        estimate (conditional indirect effect), simple_slope, se, se_slope, z,
        p_value, w, n, method.

    References
    ----------
    Preacher, Rucker & Hayes (2007), Multivariate Behavioral Research 42(1),
    185-227, p.197 and eq. (8), (13), (14).
    """
    wv = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    n = int(wv.size)
    slope = [float(a1) + float(a3) * float(wv[i]) for i in range(n)]
    eff = [float(b) * s for s in slope]
    if sa1 is None or sa3 is None or sb is None:
        se = [float("nan")] * n
        se_slope = [float("nan")] * n
    else:
        vslope = [
            float(sa1) ** 2 + 2.0 * float(sa1a3) * float(wv[i]) + float(sa3) ** 2 * float(wv[i]) ** 2
            for i in range(n)
        ]
        se_slope = [float(np.sqrt(v)) if v >= 0.0 else float("nan") for v in vslope]
        se = [
            float(np.sqrt(slope[i] ** 2 * float(sb) ** 2 + float(b) ** 2 * vslope[i]))
            for i in range(n)
        ]
    z = [eff[i] / se[i] if se[i] == se[i] and se[i] > 0.0 else float("nan") for i in range(n)]
    pv = [2.0 * (1.0 - float(_t3.normcdf(abs(z[i])))) if z[i] == z[i] else float("nan") for i in range(n)]
    if n == 1:
        return RichResult(
            payload={
                "estimate": eff[0],
                "simple_slope": slope[0],
                "se": se[0],
                "se_slope": se_slope[0],
                "z": z[0],
                "p_value": pv[0],
                "w": float(wv[0]),
                "n": 1,
                "method": "Conditional indirect effect (Preacher, Rucker & Hayes 2007)",
            }
        )
    return RichResult(
        payload={
            "estimate": float(np.mean(np.asarray(eff, dtype=float))),
            "effect": np.asarray(eff, dtype=float),
            "simple_slope": np.asarray(slope, dtype=float),
            "se": np.asarray(se, dtype=float),
            "se_slope": np.asarray(se_slope, dtype=float),
            "z": np.asarray(z, dtype=float),
            "p_value": np.asarray(pv, dtype=float),
            "w": wv,
            "n": n,
            "method": "Conditional indirect effect (Preacher, Rucker & Hayes 2007)",
        }
    )


# CANONICAL TEST
# >>> # with a3 = 0 the effect stops depending on W and reduces to a1 * b
# >>> r = conditional_indirect_effect(0.5, 0.0, 0.4, 3.0)
# >>> assert abs(r["estimate"] - 0.2) < 1e-12
# >>> assert abs(r["simple_slope"] - 0.5) < 1e-12


def cheatsheet():
    return "condie(a1, a3, b, w): conditional indirect effect + delta-method SE."
