# morie.fn -- function file (rootcoder007/morie)
"""EM for state-space parameters."""

import math

from ._richresult import RichResult

__all__ = ["em_state_space"]


def em_state_space(y, init=None, max_iter=50):
    """
    EM for state-space parameters

    Formula: E-step Kalman filter + RTS smoother (including the lag-one
    smoothed covariance); M-step in closed form.  For the scalar model

        x_t = phi x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = x_t + v_t,          v_t ~ N(0, R)

    with the smoothed sums

        S11 = sum_t (xs_t^2 + Ps_t)
        S10 = sum_t (xs_t xs_{t-1} + Pcs_t)
        S00 = sum_t (xs_{t-1}^2 + Ps_{t-1})

    the M-step is exactly

        phi = S10 / S00
        Q   = (S11 - phi S10) / n
        R   = (1/n) sum_t [(y_t - xs_t)^2 + Ps_t]

    Parameters
    ----------
    y : array-like
        Observation sequence.
    init : array-like, optional
        Starting values (phi, Q, R).  Default (0.9, var(y)/2, var(y)/2).
    max_iter : int
        Number of EM iterations.  ``0`` returns the log-likelihood at the
        starting values without moving them.

    Returns
    -------
    result : dict
        Keys: estimate (phi), phi, Q, R, loglik, loglik_path, iters, n,
        method.

    References
    ----------
    Shumway & Stoffer (1982), J. Time Series Analysis 3(4):253-264,
    doi:10.1111/j.1467-9892.1982.tb00349.x.
    """
    y = [float(v) for v in y]
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if init is None:
        m = sum(y) / n
        v = sum((t - m) ** 2 for t in y) / (n - 1) if n > 1 else 1.0
        if v <= 0.0:
            v = 1.0
        phi, Q, R = 0.9, v / 2.0, v / 2.0
    else:
        init = [float(v) for v in init]
        if len(init) != 3:
            raise ValueError("init must be (phi, Q, R)")
        phi, Q, R = init
    if Q < 0.0 or R <= 0.0:
        raise ValueError("Q must be non-negative and R positive")
    mu0 = 0.0
    Sig0 = 1.0
    path = []
    loglik = float("nan")
    for _ in range(max_iter + 1):
        # --- E-step: forward filter -------------------------------------
        xp = [0.0] * (n + 1)
        Pp = [0.0] * (n + 1)
        xf = [0.0] * (n + 1)
        Pf = [0.0] * (n + 1)
        Kg = [0.0] * (n + 1)
        xf[0], Pf[0] = mu0, Sig0
        loglik = 0.0
        for t in range(1, n + 1):
            xp[t] = phi * xf[t - 1]
            Pp[t] = phi * phi * Pf[t - 1] + Q
            S = Pp[t] + R
            Kg[t] = Pp[t] / S
            e = y[t - 1] - xp[t]
            xf[t] = xp[t] + Kg[t] * e
            Pf[t] = (1.0 - Kg[t]) * Pp[t]
            loglik += -0.5 * (math.log(2.0 * math.pi * S) + e * e / S)
        path.append(loglik)
        if len(path) > max_iter:
            break
        # --- E-step: RTS smoother ---------------------------------------
        xs = list(xf)
        Ps = list(Pf)
        J = [0.0] * (n + 1)
        for t in range(n - 1, -1, -1):
            J[t] = Pf[t] * phi / Pp[t + 1] if Pp[t + 1] > 0.0 else 0.0
            xs[t] = xf[t] + J[t] * (xs[t + 1] - xp[t + 1])
            Ps[t] = Pf[t] + J[t] * J[t] * (Ps[t + 1] - Pp[t + 1])
        Pcs = [0.0] * (n + 1)
        Pcs[n] = (1.0 - Kg[n]) * phi * Pf[n - 1]
        for t in range(n - 1, 0, -1):
            Pcs[t] = Pf[t] * J[t - 1] + J[t] * (Pcs[t + 1] - phi * Pf[t]) * J[t - 1]
        # --- M-step -----------------------------------------------------
        S11 = sum(xs[t] * xs[t] + Ps[t] for t in range(1, n + 1))
        S10 = sum(xs[t] * xs[t - 1] + Pcs[t] for t in range(1, n + 1))
        S00 = sum(xs[t - 1] * xs[t - 1] + Ps[t - 1] for t in range(1, n + 1))
        phi = S10 / S00 if S00 > 0.0 else 0.0
        Q = (S11 - phi * S10) / n
        if Q < 0.0:
            Q = 0.0
        R = sum((y[t - 1] - xs[t]) ** 2 + Ps[t] for t in range(1, n + 1)) / n
        if R <= 0.0:
            R = 1e-12
    return RichResult(payload={
        "estimate": phi,
        "phi": phi,
        "Q": Q,
        "R": R,
        "loglik": loglik,
        "loglik_path": path,
        "iters": max_iter,
        "n": n,
        "method": "EM for state-space parameters",
    })


def cheatsheet():
    return "emkfst: EM for state-space parameters"


# compact alias per ledger/NAMING.md
emstatespace = em_state_space
