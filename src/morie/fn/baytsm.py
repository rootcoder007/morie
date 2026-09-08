# morie.fn -- function file (rootcoder007/morie)
r"""The first-order dynamic linear model, filtered and smoothed.

A local-level DLM is the Bayesian statement that the series is a random
walk observed with noise. Its whole behaviour is governed by the SIGNAL
TO NOISE RATIO W/V: with W/V near zero the filter is a long-run average
and barely moves; large, it tracks each observation. West and Harrison
make that explicit through the adaptive coefficient
:math:`A_t = R_t/(R_t + V)`, which is the fraction of each forecast
error taken into the state -- and which converges to a constant, so the
filter forgets the past geometrically at a rate the ratio fixes.

Forward filtering gives the online posterior; the backward recursion
gives the smoothed states, which use future data and are therefore the
right thing to report retrospectively. Both are returned because
confusing them is the standard misreading of a state-space fit.

References
----------
West, M. and Harrison, J. (1997) *Bayesian Forecasting and Dynamic
Models*, 2nd ed., Springer, Ch. 2 (the first-order polynomial DLM, the
adaptive coefficient and its limiting behaviour) and Sec. 4.8 (the
retrospective/smoothing recursion).

Petris, G., Petrone, S. and Campagnoli, P. (2009) *Dynamic Linear Models
with R*, Springer, Ch. 2.

Durbin, J. and Koopman, S. J. (2012) *Time Series Analysis by State
Space Methods*, 2nd ed., Oxford University Press, Ch. 4.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dlm_local_level"]

_EPS = 1e-12


def dlm_local_level(y, V=1.0, W=0.1, m0=0.0, C0=1e6):
    r"""Filter and smooth the first-order polynomial DLM."""
    obs = [float(v) for v in k.vec(y)]
    n = len(obs)
    if n == 0:
        raise ValueError("baytsm: an empty series has nothing to filter")
    V = float(V)
    W = float(W)
    if V <= 0.0:
        raise ValueError("baytsm: the observation variance must be positive")
    if W < 0.0:
        raise ValueError("baytsm: the evolution variance cannot be negative")

    m, C = float(m0), float(C0)
    ms, Cs, Rs, fs, Qs, As, es = [], [], [], [], [], [], []
    loglik = 0.0
    for t in range(n):
        R = C + W                      # prior variance for the state
        f = m                          # one-step forecast
        Q = R + V                      # forecast variance
        e = obs[t] - f
        A = R / Q                      # adaptive coefficient
        m = f + A * e
        C = R - A * A * Q              # == A * V
        ms.append(m)
        Cs.append(C)
        Rs.append(R)
        fs.append(f)
        Qs.append(Q)
        As.append(A)
        es.append(e)
        loglik += -0.5 * (math.log(2.0 * math.pi * Q) + e * e / Q)

    # retrospective recursion: s_t = m_t + (C_t / R_{t+1}) (s_{t+1} - m_t)
    sm = [0.0] * n
    sC = [0.0] * n
    sm[n - 1] = ms[n - 1]
    sC[n - 1] = Cs[n - 1]
    for t in range(n - 2, -1, -1):
        B = Cs[t] / Rs[t + 1] if Rs[t + 1] > _EPS else 0.0
        sm[t] = ms[t] + B * (sm[t + 1] - ms[t])
        sC[t] = Cs[t] + B * B * (sC[t + 1] - Rs[t + 1])

    return RichResult(payload={
        "estimate": sm, "smoothed": sm, "smoothed_var": sC,
        "filtered": ms, "filtered_var": Cs,
        "forecast": fs, "forecast_var": Qs,
        "adaptive_coefficient": As, "forecast_error": es,
        "loglik": loglik, "signal_to_noise": W / V, "n": n,
        "V": V, "W": W,
        "method": "first-order polynomial DLM, forward filter and "
                  "retrospective smoother (West & Harrison 1997 Ch. 2, "
                  "Sec. 4.8)",
        "note": "the adaptive coefficient A = R/(R+V) is the fraction of "
                "each forecast error taken into the state; it converges, so "
                "the filter forgets the past geometrically at a rate W/V "
                "fixes",
    })


def cheatsheet():
    return ("baytsm: dlm_local_level(y, V, W, m0, C0) -> filtered and "
            "smoothed states of the first-order DLM (West & Harrison 1997, "
            "Bayesian Forecasting and Dynamic Models, Ch. 2)")
