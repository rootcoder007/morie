# morie.fn -- slice s03 (rootcoder007/morie)
"""Mixture cure model.

Source consulted: Sy, J. P. and Taylor, J. M. G. (2000).  Estimation in
a Cox proportional hazards cure model.  *Biometrics* 56(1), 227-236, and
Kuk, A. Y. C. and Chen, C.-H. (1992).  A mixture model combining
logistic regression with proportional hazards regression.  *Biometrika*
79(3), 531-541.  The survival function is

    S(t | X, Z) = (1 - pi(Z)) + pi(Z) S_0(t | X)

where pi(Z) = expit(gamma' Z) is the probability of being *uncured* --
susceptible -- and S_0 is the survival of a susceptible subject.  Note
the orientation: as t -> infinity, S -> 1 - pi, so 1 - pi is the cure
fraction.  Both papers are paywalled; the mixture is quoted in its
standard published form.

Estimation is by the EM algorithm of Sy and Taylor: the E-step imputes
the susceptibility indicator of each censored subject by its conditional
probability, the M-step refits the logistic incidence model and the
Kaplan-Meier latency model on those weights.  The zero-tail constraint
Sy and Taylor impose -- S_0 is set to zero beyond the largest
*uncensored* time -- is applied, since without it the cure fraction is
not identified.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["cure_model"]


def cure_model(time, event, X=None, Z=None, max_iter=200, tol=1e-12):
    """EM fit of a logistic-incidence / Kaplan-Meier-latency cure model.

    Returns
    -------
    estimate : the cure fraction 1 - mean(pi)
    cure_fraction : same as estimate
    pi       : susceptibility probability per subject
    gamma    : logistic incidence coefficients
    S0       : the latency survival at each distinct event time
    """
    t = k.vec(time)
    e = k.vec(event)
    n = len(t)
    Zd = k.design(Z, n)
    times = sorted(set([t[i] for i in range(n) if e[i] > 0.5]))
    tmax = max(times) if times else float("inf")
    w = [1.0 if e[i] > 0.5 else 0.5 for i in range(n)]
    gam = [0.0] * len(Zd[0])
    S0 = [1.0] * len(times)
    for _ in range(int(max_iter)):
        gam = k.logit_irls(Zd, w, 40)
        pi = [k.sigmoid(v) for v in k.matvec(Zd, gam)]
        # weighted Kaplan-Meier for the susceptible sub-population
        surv = 1.0
        S0 = []
        for tt in times:
            d = 0.0
            r = 0.0
            for i in range(n):
                if t[i] >= tt:
                    r += w[i]
                if abs(t[i] - tt) < 1e-12 and e[i] > 0.5:
                    d += 1.0
            surv *= (1.0 - d / r) if r > 0.0 else 1.0
            S0.append(surv)
        neww = []
        for i in range(n):
            if e[i] > 0.5:
                neww.append(1.0)
                continue
            s = 1.0
            for j in range(len(times)):
                if times[j] <= t[i]:
                    s = S0[j]
            if t[i] >= tmax:
                s = 0.0
            num = pi[i] * s
            den = (1.0 - pi[i]) + num
            neww.append(num / den if den > 0.0 else 0.0)
        delta = 0.0
        for i in range(n):
            d = abs(neww[i] - w[i])
            if d > delta:
                delta = d
        w = neww
        if delta < tol:
            break
    pi = [k.sigmoid(v) for v in k.matvec(Zd, gam)]
    cure = 1.0 - k.mean(pi)
    return RichResult(
        title="Mixture cure model",
        summary_lines=[("cure fraction", cure)],
        payload={
            "estimate": cure,
            "cure_fraction": cure,
            "pi": pi,
            "gamma": gam,
            "S0": S0,
            "times": times,
            "weights": w,
            "n": n,
            "method": "Sy and Taylor (2000) EM cure model with the zero-tail constraint",
        },
    )


def cheatsheet():
    return "sscure: Mixture cure model"
