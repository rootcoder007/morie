# morie.fn -- function file (rootcoder007/morie)
"""LP-DiD: local-projection difference-in-differences with clean controls.

Source opened: Dube, A., Girardi, D., Jorda, O. and Taylor, A. M.
(2023).  A local projections approach to difference-in-differences.
NBER Working Paper 31184, page 9, equation (8); published as *Journal of
Applied Econometrics* 40(7), 741-758 (2025), doi:10.1002/jae.70000.
The specification is, for each horizon h,

    y_{i,t+h} - y_{i,t-1} = beta_h dD_{it} + delta_{th} + e_{ith}

estimated only on observations that are either

    newly treated   dD_{it} = 1        or
    clean control   D_{i,t+h} = 0.

The clean-control restriction is the whole point: it removes the
already-treated units from the comparison group, and with them the
negative weights that make the dynamic two-way fixed effects
specification uninterpretable under heterogeneous effects.

Each horizon's contrast is estimated by the doubly robust moment of
Sant'Anna and Zhao (2020), eq. (2.6), rather than by OLS, which is the
"DR weights" the interface asks for; with no covariates the DR moment is
the difference of the two group means, so beta_0 must equal the raw
newly-treated minus clean-control mean change -- the degenerate check.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_lp_did"]


def dr_lp_did(y, D, unit, time, horizon=3, X=None):
    """LP-DiD impulse response beta_h for h = 0 .. horizon.

    Parameters
    ----------
    y : array-like
        Outcome, long format, one entry per unit-period.
    D : array-like
        Binary treatment indicator, absorbing or not.
    unit, time : array-like
        Unit and period identifiers.
    horizon : int
        Largest horizon reported; must be non-negative.
    X : 2-D array-like, optional
        Baseline covariates, one row per unit-period.

    Returns
    -------
    result : dict
        Keys: estimate (beta at h = 0), horizons, beta, se, n_cells, n.

    References
    ----------
    Dube, Girardi, Jorda & Taylor (2023), NBER WP 31184, eq. (8);
    J. Applied Econometrics 40(7):741-758 (2025), doi:10.1002/jae.70000.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
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
    H = int(horizon)
    if H < 0:
        raise ValueError("horizon must be non-negative")
    Xr = k.mat(X) if X is not None else None
    val, dof, xof = {}, {}, {}
    for i in range(n):
        val[(u[i], t[i])] = yv[i]
        dof[(u[i], t[i])] = dv[i]
        if Xr is not None:
            xof[(u[i], t[i])] = Xr[i]
    units = []
    for x in u:
        if x not in units:
            units.append(x)
    per = sorted(set(t))
    hs = list(range(0, H + 1))
    beta, sev, ncell = [], [], []
    for h in hs:
        dys, ds, xs = [], [], []
        for j in range(1, len(per)):
            tc, tp = per[j], per[j - 1]
            if j + h >= len(per):
                continue
            th = per[j + h]
            for z in units:
                if (z, th) not in val or (z, tp) not in val:
                    continue
                if (z, tc) not in dof or (z, th) not in dof:
                    continue
                new = (dof[(z, tc)] >= 0.5 and dof[(z, tp)] < 0.5)
                clean = (dof[(z, th)] < 0.5)
                if not (new or clean):
                    continue
                dys.append(val[(z, th)] - val[(z, tp)])
                ds.append(1.0 if new else 0.0)
                if Xr is not None:
                    xs.append(xof[(z, tc)])
        ncell.append(float(len(dys)))
        if len(dys) < 3 or sum(ds) <= 0.0 or sum(ds) >= len(ds):
            beta.append(float("nan"))
            sev.append(float("nan"))
            continue
        f = k.drdid_panel(dys, ds, xs if Xr is not None else None)
        beta.append(f["tau"])
        sev.append(f["se"])
    return RichResult(
        title="LP-DiD",
        summary_lines=[("horizons", len(hs))],
        payload={
            "estimate": beta[0] if beta else float("nan"),
            "horizons": hs,
            "beta": beta,
            "se": sev,
            "n_cells": ncell,
            "n": n,
            "method": "DR-DiD via local projection",
        },
    )


def cheatsheet():
    return "drlp1: DR-DiD via local projection"


# compact alias per ledger/NAMING.md
drlpdid = dr_lp_did
