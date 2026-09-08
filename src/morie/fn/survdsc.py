# morie.fn -- slice s03 (rootcoder007/morie)
"""Discrete-time survival with a complementary log-log link.

Source consulted: Prentice, R. L. and Gloeckler, L. A. (1978).
Regression analysis of grouped survival data with application to breast
cancer data.  *Biometrics* 34(1), 57-67.  Grouping a proportional
hazards model into intervals gives exactly the complementary log-log
model

    log( -log( 1 - h_t(x) ) ) = alpha_t + beta' x

with h_t the discrete hazard in interval t, and beta identical to the
continuous-time Cox coefficient -- which is why this link, and not the
logit, is the one that makes the discrete model a *grouped* proportional
hazards model.  The 1978 paper is paywalled; the link and that
identification are quoted in their standard published form.

The model is fitted by Newton-Raphson on the person-period likelihood,
one row per subject per interval at risk.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["discrete_time_survival"]


def discrete_time_survival(time_discrete, event, X=None, max_iter=100,
                           tol=1e-12, ridge=1e-8):
    """Grouped proportional hazards by complementary log-log regression.

    Returns
    -------
    estimate : the first regression coefficient beta_1
    alpha    : the interval baseline terms
    beta     : the covariate coefficients
    hazard   : the fitted baseline hazard per interval
    loglik
    """
    t = [int(x) for x in k.vec(time_discrete)]
    e = k.vec(event)
    n = len(t)
    Xr = k.mat(X) if X is not None else [[] for _ in range(n)]
    p = len(Xr[0]) if Xr and Xr[0] else 0
    ivals = sorted(set(t))
    T = len(ivals)
    rows = []
    ys = []
    for i in range(n):
        for j in range(T):
            if ivals[j] > t[i]:
                break
            d = [1.0 if a == j else 0.0 for a in range(T)]
            rows.append(d + list(Xr[i]))
            ys.append(1.0 if (ivals[j] == t[i] and e[i] > 0.5) else 0.0)
    m = len(rows)
    q = T + p
    beta = [0.0] * q
    ll = float("-inf")
    for _ in range(int(max_iter)):
        gr = [0.0] * q
        H = [[0.0] * q for _ in range(q)]
        ll = 0.0
        for r in range(m):
            eta = 0.0
            for a in range(q):
                eta += rows[r][a] * beta[a]
            ee = math.exp(eta) if eta < 700.0 else math.exp(700.0)
            h = 1.0 - math.exp(-ee)
            if h < 1e-12:
                h = 1e-12
            if h > 1.0 - 1e-12:
                h = 1.0 - 1e-12
            ll += ys[r] * math.log(h) + (1.0 - ys[r]) * math.log(1.0 - h)
            dh = ee * math.exp(-ee)
            s = (ys[r] / h - (1.0 - ys[r]) / (1.0 - h)) * dh
            wgt = dh * dh / (h * (1.0 - h))
            for a in range(q):
                gr[a] += rows[r][a] * s
                for b in range(q):
                    H[a][b] += rows[r][a] * wgt * rows[r][b]
        step = k.ridgesolve(H, gr, ridge)
        mx = 0.0
        for a in range(q):
            beta[a] += step[a]
            if abs(step[a]) > mx:
                mx = abs(step[a])
        if mx < tol:
            break
    haz = [1.0 - math.exp(-math.exp(beta[j])) for j in range(T)]
    return RichResult(
        title="Discrete-time survival (cloglog)",
        summary_lines=[("intervals", T), ("log-lik", ll)],
        payload={
            "estimate": beta[T] if p else float("nan"),
            "alpha": beta[:T],
            "beta": beta[T:],
            "hazard": haz,
            "loglik": ll,
            "intervals": ivals,
            "n": n,
            "n_person_periods": m,
            "method": "Grouped proportional hazards by complementary log-log (Prentice and Gloeckler 1978)",
        },
    )


def cheatsheet():
    return "survdsc: Discrete-time survival via complementary log-log"
