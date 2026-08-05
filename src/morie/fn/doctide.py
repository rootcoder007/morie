# morie.fn -- function file (rootcoder007/morie)
"""de Chaisemartin and D'Haultfoeuille (2020) DID_M estimator.

Source opened: de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
Two-way fixed effects estimators with heterogeneous treatment effects.
*American Economic Review* 110(9), 2964-2996; working paper
arXiv:1803.08807, page 16 (rendered as an image, minus signs checked),
which defines, for every period t >= 2,

    DID_{+,t} = sum_{g: D_gt = 1, D_g,t-1 = 0} (N_gt / N_{1,0,t}) dY_gt
              - sum_{g: D_gt = D_g,t-1 = 0} (N_gt / N_{0,0,t}) dY_gt
    DID_{-,t} = sum_{g: D_gt = D_g,t-1 = 1} (N_gt / N_{1,1,t}) dY_gt
              - sum_{g: D_gt = 0, D_g,t-1 = 1} (N_gt / N_{0,1,t}) dY_gt

with dY_gt = Y_gt - Y_g,t-1, and the aggregate

    DID_M = sum_{t=2}^{T} ( N_{1,0,t}/N_S DID_{+,t}
                          + N_{0,1,t}/N_S DID_{-,t} ),   N_S = sum_t (N_{1,0,t} + N_{0,1,t}).

The paper is explicit that DID_{+,t} is set to zero when either of its
two donor sets is empty, and likewise for DID_{-,t}; that convention is
reproduced here rather than propagating NaN.  Each (unit, period) row
carries weight N_gt = 1, the balanced-panel case.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["de_chaisemartin_dhaultfoeuille"]


def de_chaisemartin_dhaultfoeuille(y, D, unit, time):
    """DID_M, robust to heterogeneous and dynamic treatment effects.

    Parameters
    ----------
    y : array-like
        Outcome, long format, one entry per unit-period.
    D : array-like
        Binary treatment indicator, same length as ``y``.
    unit, time : array-like
        Unit and period identifiers, same length as ``y``.

    Returns
    -------
    result : dict
        Keys: estimate (DID_M), periods, did_plus, did_minus, n10, n01,
        n_switch, n_units, n.

    References
    ----------
    de Chaisemartin, C. and D'Haultfoeuille, X. (2020). American
    Economic Review 110(9):2964-2996, eq. of Section 3 p. 16 of
    arXiv:1803.08807.
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    u = [str(x) for x in unit]
    t = [float(x) for x in time]
    if not (len(dv) == n and len(u) == n and len(t) == n):
        raise ValueError("y, D, unit and time must have the same length")
    for x in dv:
        if x != 0.0 and x != 1.0:
            raise ValueError("D must be binary 0/1")
    yof = {}
    dof = {}
    for i in range(n):
        yof[(u[i], t[i])] = yv[i]
        dof[(u[i], t[i])] = dv[i]
    units = []
    for x in u:
        if x not in units:
            units.append(x)
    per = sorted(set(t))
    if len(per) < 2:
        raise ValueError("DID_M needs at least two periods")
    dplus = []
    dminus = []
    n10 = []
    n01 = []
    for j in range(1, len(per)):
        tc, tp = per[j], per[j - 1]
        sw_in, st_0, st_1, sw_out = [], [], [], []
        for g in units:
            if (g, tc) not in yof or (g, tp) not in yof:
                continue
            dy = yof[(g, tc)] - yof[(g, tp)]
            a, b = dof[(g, tc)], dof[(g, tp)]
            if a == 1.0 and b == 0.0:
                sw_in.append(dy)
            elif a == 0.0 and b == 0.0:
                st_0.append(dy)
            elif a == 1.0 and b == 1.0:
                st_1.append(dy)
            else:
                sw_out.append(dy)
        dp = (k.mean(sw_in) - k.mean(st_0)) if (sw_in and st_0) else 0.0
        dm = (k.mean(st_1) - k.mean(sw_out)) if (st_1 and sw_out) else 0.0
        dplus.append(dp)
        dminus.append(dm)
        n10.append(float(len(sw_in)))
        n01.append(float(len(sw_out)))
    ns = sum(n10) + sum(n01)
    if ns <= 0.0:
        raise ValueError("no group switches treatment: DID_M is not defined")
    did_m = 0.0
    for j in range(len(dplus)):
        did_m += (n10[j] / ns) * dplus[j] + (n01[j] / ns) * dminus[j]
    return RichResult(
        title="DID_M",
        summary_lines=[("switching cells", ns)],
        payload={
            "estimate": did_m,
            "periods": per[1:],
            "did_plus": dplus,
            "did_minus": dminus,
            "n10": n10,
            "n01": n01,
            "n_switch": ns,
            "n_units": len(units),
            "n": n,
            "method": "de Chaisemartin-D'Haultfoeuille heterogeneous DID",
        },
    )


def cheatsheet():
    return "doctide: de Chaisemartin-D'Haultfoeuille heterogeneous DID"
