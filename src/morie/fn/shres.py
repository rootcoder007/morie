# morie.fn -- slice s03 (rootcoder007/morie)
"""Schoenfeld residuals.

Source consulted: Schoenfeld, D. (1982).  Partial residuals for the
proportional hazards regression model.  *Biometrika* 69(1), 239-241.
For the i-th event time and the k-th covariate the residual is

    r_ki = x_ki - xbar_k( t_i , betahat )

where xbar_k is the risk-set average of covariate k weighted by the
fitted risk scores,

    xbar_k(t, beta) = sum_(j in R(t)) x_jk exp(beta' x_j)
                      / sum_(j in R(t)) exp(beta' x_j)

so the residual is defined only at *event* times, one per event.  The
1982 note is paywalled; both expressions are quoted in their standard
published form.

The scaled residuals of Grambsch and Therneau (1994), *Biometrika*
81(3), 515-526 -- r* = d V^(-1) r + betahat, with d the number of events
-- are returned as well, because it is their correlation with time, not
the raw residuals', that tests proportional hazards.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["schoenfeld_residual"]


def schoenfeld_residual(time, event, X, beta=None):
    """Raw and scaled Schoenfeld residuals, and the PH correlation test.

    Returns
    -------
    estimate : the correlation of the first scaled residual with time
    residuals : raw residuals, one row per event
    scaled    : scaled residuals
    event_times
    rho       : correlation with time per covariate
    """
    t = k.vec(time)
    e = k.vec(event)
    Xm = k.mat(X)
    n = len(t)
    p = len(Xm[0]) if n else 0
    b = k.vec(beta) if beta is not None else [0.0] * p
    order = sorted(range(n), key=lambda i: (t[i], i))
    ets = []
    res = []
    V = [[0.0] * p for _ in range(p)]
    for pos in range(n):
        i = order[pos]
        if e[i] < 0.5:
            continue
        risk = [j for j in order if t[j] >= t[i]]
        wsum = 0.0
        xbar = [0.0] * p
        for j in risk:
            eta = 0.0
            for a in range(p):
                eta += b[a] * Xm[j][a]
            wj = math.exp(eta)
            wsum += wj
            for a in range(p):
                xbar[a] += wj * Xm[j][a]
        xbar = [x / wsum if wsum > 0.0 else 0.0 for x in xbar]
        for j in risk:
            eta = 0.0
            for a in range(p):
                eta += b[a] * Xm[j][a]
            wj = math.exp(eta) / wsum if wsum > 0.0 else 0.0
            for a in range(p):
                for c in range(p):
                    V[a][c] += wj * (Xm[j][a] - xbar[a]) * (Xm[j][c] - xbar[c])
        ets.append(t[i])
        res.append([Xm[i][a] - xbar[a] for a in range(p)])
    d = len(res)
    Vm = [[V[a][c] / d if d else 0.0 for c in range(p)] for a in range(p)]
    scaled = []
    for r in res:
        s = k.ridgesolve(Vm, r, 1e-10)
        scaled.append([s[a] / d + b[a] if d else b[a] for a in range(p)])
    rho = []
    for a in range(p):
        col = [scaled[i][a] for i in range(d)]
        rho.append(k.corr(ets, col) if d > 1 else float("nan"))
    return RichResult(
        title="Schoenfeld residuals",
        summary_lines=[("events", d)],
        payload={
            "estimate": rho[0] if rho else float("nan"),
            "residuals": res,
            "scaled": scaled,
            "event_times": ets,
            "rho": rho,
            "V": Vm,
            "n_events": d,
            "method": "Schoenfeld (1982) residuals with the Grambsch-Therneau (1994) scaling",
        },
    )


def cheatsheet():
    return "shres: Unscaled Schoenfeld residuals for PH assumption"
