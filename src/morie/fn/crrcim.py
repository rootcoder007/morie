# morie.fn -- function file (rootcoder007/morie)
"""Cumulative incidence function for competing risks."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["cumulative_incidence"]


def aalen_johansen(time, event_type, cause=1):
    """Aalen-Johansen estimator of F_k(t) plus the overall KM survival.

    F_k(t) = sum_{u <= t} S(u-) dN_k(u) / Y(u).  The S(u-) factor is the
    ALL-CAUSE Kaplan-Meier: leaving it out gives 1 - KM computed on the
    cause alone, which over-states the incidence whenever a competing
    event can happen first.
    """
    t = core.vec(time)
    n = len(t)
    if n == 0:
        raise ValueError("empty input: time has no observations")
    ev = core.vec(event_type)
    if len(ev) != n:
        raise ValueError("time and event_type must have the same length")
    if any(v < 0.0 for v in t):
        raise ValueError("times must be non-negative")
    cause = float(cause)
    order = sorted(range(n), key=lambda i: (t[i], i))
    ts = [t[i] for i in order]
    es = [ev[i] for i in order]
    times, F, S, atrisk, dk = [], [], [], [], []
    surv = 1.0
    cif = 0.0
    i = 0
    while i < n:
        u = ts[i]
        j = i
        d_all = 0
        d_k = 0
        while j < n and ts[j] == u:
            if es[j] != 0.0:
                d_all += 1
                if es[j] == cause:
                    d_k += 1
            j += 1
        Y = n - i
        if d_all > 0:
            cif += surv * d_k / Y
            surv *= (1.0 - d_all / float(Y))
            times.append(u)
            F.append(cif)
            S.append(surv)
            atrisk.append(float(Y))
            dk.append(float(d_k))
        i = j
    return times, F, S, atrisk, dk, n


def cumulative_incidence(time, event_type, cause=1):
    """
    Cumulative incidence function under competing risks

    Formula: F_k(t) = integral S(u-) lambda_k(u) du

    The Aalen-Johansen estimator.  With a single cause and no competing
    event it collapses to exactly 1 - KM(t), which is the identity used
    to check it; with competing events it stays below that, because a
    subject who fails of another cause can never fail of this one.

    Parameters
    ----------
    time : array-like
        Follow-up time per subject.
    event_type : array-like
        0 for censored, otherwise the cause label.
    cause : int
        Cause of interest.

    Returns
    -------
    result : dict
        Keys: estimate (F_k at the last event time), time, cif, surv,
        n_risk, n_event, n.

    References
    ----------
    Kalbfleisch & Prentice (2002), The Statistical Analysis of Failure
    Time Data, 2nd ed., Wiley, section 8.2.
    Aalen & Johansen (1978), Scand. J. Statist. 5(3):141-150.
    """
    times, F, S, Y, dk, n = aalen_johansen(time, event_type, cause)
    return RichResult(payload={
        "estimate": F[-1] if F else 0.0,
        "time": times,
        "cif": F,
        "surv": S,
        "n_risk": Y,
        "n_event": dk,
        "n": n,
        "method": "Aalen-Johansen cumulative incidence function",
    })


def cheatsheet():
    return "crrcim: cumulative incidence function (Aalen-Johansen)"


# compact alias per ledger/NAMING.md
cumulativeincidence = cumulative_incidence
