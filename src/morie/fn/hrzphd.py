# morie.fn -- function file (rootcoder007/morie)
"""Proportional hazards with gamma heterogeneity, Y observed in intervals.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.2.3, pages 208-209 (volume
[Pages 189-232], read as rendered page images).  When Y is observed only
through the interval (y_{j-1}, y_j] that contains it, Lambda_0 is
identified at the K boundaries only, so the model of Section 6.2.2 -- a
proportional hazard with a gamma frailty of variance theta -- becomes
finite-dimensional.  With y_0 = 0 the book gives, for 1 <= j <= K,

    P(y_{j-1} < Y <= y_j | X = x)
        = [1 + theta Lambda_0(y_{j-1}) e^{-x b}]^{-1/theta}
        - [1 + theta Lambda_0(y_j)     e^{-x b}]^{-1/theta}
    P(Y > y_K | X = x) = [1 + theta Lambda_0(y_K) e^{-x b}]^{-1/theta}

and then the log likelihood of a random sample of (Y, X).

BOOK NOTE (sign).  The two displayed probabilities carry the exponent
-1/theta; the log likelihood printed immediately below them on p. 208
carries +1/t in all three places.  With A_{j-1} < A_j the +1/t reading
makes the bracketed difference negative and its logarithm undefined,
while the -1/t reading is exactly the probability displayed two lines
above.  The exponent is -1/t; that is what is implemented.  This is the
same dropped minus this book is already known for.

Estimation is by direct maximisation of the log likelihood over
theta > 0, beta, and the K jumps of Lambda_0, using theta = exp(tau) and
Lambda_0(y_j) = sum_{l <= j} exp(c_l) so the parameters are
unconstrained and Lambda_0 is automatically nonnegative and increasing.
The maximiser is cyclic coordinate golden-section search from a fixed
start: deterministic, no random restarts.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_ph_discrete_obs"]

_GR = 0.6180339887498949


def horowitz_ph_discrete_obs(t_discrete, x, event=None, K=None, cycles=40, gs_iter=48):
    """Interval-censored gamma-frailty proportional hazards.

    Parameters
    ----------
    t_discrete : array-like
        For an observed event, the index j in 1, ..., K of the interval
        (y_{j-1}, y_j] that contains Y.  Ignored where event = 0.
    x : array-like
        n-by-p covariate matrix WITHOUT an intercept: Lambda_0 absorbs it.
    event : array-like, optional
        1 if Y fell in interval t_discrete, 0 if Y > y_K.  All ones if
        omitted, in which case every observation is an interval event.
    K : int, optional
        Number of intervals; the largest observed index if omitted.
    cycles, gs_iter :
        Coordinate-descent controls.

    Returns
    -------
    beta_hat : the coefficients, in the books exp(-x b) sign convention
    h_j_hat  : the K jumps of Lambda_0
    theta_hat: the gamma frailty variance
    """
    jj = core.vec(t_discrete)
    XX = core.mat(x)
    n = len(jj)
    if n == 0:
        raise ValueError("horowitz_ph_discrete_obs: t_discrete is empty")
    if len(XX) != n:
        raise ValueError("horowitz_ph_discrete_obs: x has a different number of rows than t_discrete")
    p = len(XX[0])
    if event is None:
        ev = [1.0] * n
    else:
        ev = core.vec(event)
        if len(ev) != n:
            raise ValueError("horowitz_ph_discrete_obs: event has a different length than t_discrete")
    kk = 0
    for i in range(n):
        if ev[i] != 0.0:
            v = int(jj[i])
            if v < 1 or v != jj[i]:
                raise ValueError("horowitz_ph_discrete_obs: interval indices must be integers >= 1")
            if v > kk:
                kk = v
    if K is not None:
        kk = int(K)
    if kk < 1:
        raise ValueError("horowitz_ph_discrete_obs: no interval events, K is undefined")

    def cumA(c):
        A = [0.0] * (kk + 1)
        s = 0.0
        for l in range(kk):
            s += math.exp(min(max(c[l], -300.0), 300.0))
            A[l + 1] = s
        return A

    def negll(par):
        tau = par[0]
        b = par[1:1 + p]
        A = cumA(par[1 + p:])
        th = math.exp(min(max(tau, -30.0), 30.0))
        tot = 0.0
        for i in range(n):
            e = 0.0
            for k in range(p):
                e += XX[i][k] * b[k]
            w = math.exp(min(max(-e, -300.0), 300.0))
            if ev[i] != 0.0:
                j = int(jj[i])
                lo = (1.0 + th * A[j - 1] * w) ** (-1.0 / th)
                hi = (1.0 + th * A[j] * w) ** (-1.0 / th)
                d = lo - hi
                tot += math.log(d) if d > 1e-300 else -1e300
            else:
                tot += (-1.0 / th) * math.log(1.0 + th * A[kk] * w)
        return -tot

    par = [0.0] * (1 + p + kk)
    for l in range(kk):
        par[1 + p + l] = -1.0
    cur = negll(par)
    for _ in range(int(cycles)):
        moved = 0.0
        for c in range(len(par)):
            lo = par[c] - 2.0
            hi = par[c] + 2.0
            a1 = hi - _GR * (hi - lo)
            a2 = lo + _GR * (hi - lo)
            q = list(par)
            q[c] = a1
            f1 = negll(q)
            q[c] = a2
            f2 = negll(q)
            for _ in range(int(gs_iter)):
                if f1 < f2:
                    hi = a2
                    a2 = a1
                    f2 = f1
                    a1 = hi - _GR * (hi - lo)
                    q[c] = a1
                    f1 = negll(q)
                else:
                    lo = a1
                    a1 = a2
                    f1 = f2
                    a2 = lo + _GR * (hi - lo)
                    q[c] = a2
                    f2 = negll(q)
            newv = 0.5 * (lo + hi)
            q[c] = newv
            fv = negll(q)
            if fv < cur:
                moved = max(moved, abs(newv - par[c]))
                par[c] = newv
                cur = fv
        if moved < 1e-10:
            break
    th = math.exp(min(max(par[0], -30.0), 30.0))
    beta = list(par[1:1 + p])
    A = cumA(par[1 + p:])
    jumps = [A[l + 1] - A[l] for l in range(kk)]
    # cell probabilities for the first observation: these must sum to one
    e = 0.0
    for k in range(p):
        e += XX[0][k] * beta[k]
    w = math.exp(min(max(-e, -300.0), 300.0))
    cells = []
    for j in range(1, kk + 1):
        cells.append((1.0 + th * A[j - 1] * w) ** (-1.0 / th) - (1.0 + th * A[j] * w) ** (-1.0 / th))
    cells.append((1.0 + th * A[kk] * w) ** (-1.0 / th))
    return RichResult(
        title="Proportional hazards with discrete observations",
        summary_lines=[("n", n), ("K", kk), ("theta", th)],
        payload={
            "estimate": th,
            "theta_hat": th,
            "beta_hat": beta,
            "h_j_hat": jumps,
            "Lambda0": A[1:],
            "loglik": -cur,
            "cell_probs": cells,
            "K": kk,
            "n": n,
            "method": "Horowitz (2009) Sec. 6.2.3 pp.208-209, interval likelihood with exponent -1/theta",
        },
    )


def cheatsheet():
    return "hrzphd: Proportional hazards with discrete observations"
