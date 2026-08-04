# morie.fn -- k02 batch (rootcoder007/morie)
"""Adjusted indirect comparison (the Bucher method).

Source consulted: Bucher, H.C., Guyatt, G.H., Griffith, L.E. and Walter, S.D.
(1997), The results of direct and indirect treatment comparisons in
meta-analysis of randomized controlled trials, *Journal of Clinical
Epidemiology* 50(6), 683-691.  With A the common comparator,

    d_BC = d_AB - d_AC,   Var(d_BC) = Var(d_AB) + Var(d_AC)

because the two direct comparisons come from disjoint sets of trials.  When a
direct estimate of BC is also supplied the function returns the
inconsistency ``d_direct - d_indirect`` with its variance and Wald test, which
is the standard node-splitting check for a two-arm network.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02p2z, k02z

from ._richresult import RichResult

__all__ = ["ma_network_indirect"]


def ma_network_indirect(d_ab, v_ab, d_ac, v_ac, d_bc=None, v_bc=None, level=0.95):
    """Bucher adjusted indirect comparison, with an optional consistency test.

    Parameters
    ----------
    d_ab, v_ab : float
        Pooled effect of B versus A and its variance.
    d_ac, v_ac : float
        Pooled effect of C versus A and its variance.
    d_bc, v_bc : float, optional
        Direct estimate of B versus C, if any.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    RichResult
        estimate (indirect d_BC), se, variance, ci_lower, ci_upper, z,
        p_value, inconsistency, se_inconsistency, p_inconsistency, n, method.
    """
    d = float(d_ab) - float(d_ac)
    var = float(v_ab) + float(v_ac)
    se = float(np.sqrt(var))
    crit = k02z(0.5 + 0.5 * float(level))
    z = d / se
    inc = None
    sei = None
    pi = None
    if d_bc is not None and v_bc is not None:
        inc = float(d_bc) - d
        sei = float(np.sqrt(float(v_bc) + var))
        pi = float(k02p2z(inc / sei))
    return RichResult(
        payload={
            "estimate": d,
            "se": se,
            "variance": var,
            "ci_lower": float(d - crit * se),
            "ci_upper": float(d + crit * se),
            "z": float(z),
            "p_value": float(k02p2z(z)),
            "inconsistency": inc,
            "se_inconsistency": sei,
            "p_inconsistency": pi,
            "n": 2 if d_bc is None else 3,
            "method": "Bucher adjusted indirect comparison (Bucher, Guyatt, Griffith & Walter 1997)",
        }
    )


# CANONICAL TEST
# >>> r = ma_network_indirect(0.40, 0.02, 0.15, 0.03, d_bc=0.20, v_bc=0.04)
# >>> assert abs(r["estimate"] - 0.25) < 1e-15
# >>> assert abs(r["variance"] - 0.05) < 1e-15
# >>> assert abs(r["inconsistency"] + 0.05) < 1e-15


def cheatsheet():
    return "manmar(d_ab, v_ab, d_ac, v_ac): Bucher adjusted indirect comparison."


manetworkindirect = ma_network_indirect
