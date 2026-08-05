# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Pointwise confidence interval for a Kaplan-Meier curve (Greenwood).

Greenwood (1926), "The natural duration of cancer", Reports on Public
Health and Medical Subjects 33:1-26, His Majesty's Stationery Office,
gives the variance of the product-limit estimator,

    Var[S(t)] = S(t)^2 * sum_{t_j <= t} d_j / ( n_j (n_j - d_j) ),

with n_j the number at risk and d_j the number of events at t_j.  The
plain (linear) pointwise interval is S(t) +/- z_{1-alpha/2} sqrt(Var),
truncated to [0, 1] because the linear scale can leave the unit
interval near the tails.  The curve is rebuilt from the risk table so
that the same fit object can be reused by the simultaneous band.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["km_pointwise_ci"]


def _risk_table(fit):
    if fit is None:
        raise ValueError("km_pointwise_ci: fit is empty")
    t = core.vec(fit["time"])
    nr = core.vec(fit["n_risk"])
    d = core.vec(fit["n_event"])
    m = len(t)
    if m == 0:
        raise ValueError("km_pointwise_ci: fit has no event times")
    if len(nr) != m or len(d) != m:
        raise ValueError("km_pointwise_ci: time, n_risk and n_event have different lengths")
    for j in range(m):
        if nr[j] <= 0:
            raise ValueError("km_pointwise_ci: n_risk must be positive")
        if d[j] < 0 or d[j] > nr[j]:
            raise ValueError("km_pointwise_ci: n_event must lie between 0 and n_risk")
    return t, nr, d, m


def km_pointwise_ci(fit, alpha):
    """Greenwood pointwise confidence interval for S(t).

    Parameters
    ----------
    fit : mapping
        Risk table with keys ``time``, ``n_risk`` and ``n_event``, one
        entry per distinct event time, in increasing time order.
    alpha : float
        Two-sided error rate, e.g. 0.05.
    """
    t, nr, d, m = _risk_table(fit)
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("km_pointwise_ci: alpha must lie in (0, 1)")
    z = core.qnorm(1.0 - a / 2.0)
    S = []
    sig2 = []
    s = 1.0
    v = 0.0
    for j in range(m):
        s *= 1.0 - d[j] / nr[j]
        if nr[j] > d[j]:
            v += d[j] / (nr[j] * (nr[j] - d[j]))
        else:
            v = float("inf")
        S.append(s)
        sig2.append(v)
    se = [S[j] * math.sqrt(sig2[j]) if sig2[j] != float("inf") else float("nan") for j in range(m)]
    lo = []
    hi = []
    for j in range(m):
        if se[j] != se[j]:
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue
        l = S[j] - z * se[j]
        u = S[j] + z * se[j]
        lo.append(0.0 if l < 0.0 else l)
        hi.append(1.0 if u > 1.0 else u)
    return RichResult(
        title="Kaplan-Meier pointwise CI (Greenwood)",
        summary_lines=[("times", m), ("S(last)", S[-1]), ("alpha", a)],
        payload={
            "estimate": S[-1],
            "time": t,
            "surv": S,
            "se": se,
            "sigma2": sig2,
            "lower": lo,
            "upper": hi,
            "z": z,
            "alpha": a,
            "n_times": float(m),
            "n_risk_start": nr[0],
            "n": m,
            "method": "S(t) +/- z sqrt(S^2 sum d/(n(n-d))), Greenwood (1926)",
        },
    )


def cheatsheet():
    return "kpmnci: KM pointwise CI via Greenwood"


# compact alias per ledger/NAMING.md
kmpointwiseci = km_pointwise_ci
