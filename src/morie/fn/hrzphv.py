# morie.fn -- function file (rootcoder007/morie)
"""Proportional hazards with a gamma frailty, by the EM algorithm.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.2.2, pages 205-208 (volume
[Pages 189-232], read as rendered page images).  With Z = exp(-V) gamma
with mean 1 and unknown variance theta (p. 205), the marginal density of
Y is (6.39), p_y(y) = lambda_0(y) / [1 + theta Lambda_0(y)]^{1 + 1/theta},
and the log likelihood over the jumps of Lambda_0 is (6.40).  Nielsen et
al. (1992) and Petersen et al. (1996) replace the (n+1)-dimensional
maximisation by an EM algorithm: (6.41) is the M step for Lambda_0,
(6.42) the E step

    E(Z_nj | Y_1, ..., Y_n) = (1 + theta_n) / (1 + theta_n Lambda_n0(Y_j)),

and a one-dimensional maximisation over theta closes the cycle.  Steps 1
to 4' on p. 208 give the version with covariates: everywhere Z_nj
appears it is replaced by Z_nj exp(-X_j' b_n), and Lambda_n0 by
Lambda_n0 exp(-X_i' b_n).

TWO BOOK NOTES, both settled against equations on the facing pages.

1. (6.41) prints the denominator as sum_{j in R(Y_i)} exp(-Z_nj).  But
   Z = exp(-V) on p. 205 is already the frailty multiplier, and (6.42)
   returns a posterior mean of that multiplier, not of a log frailty; and
   step 4' asks for "Z_nj exp(-X_j b)" in place of Z_nj, which under the
   printed reading would be exp(-Z_nj exp(-X_j b)).  The consistent
   reading, and the one used here, is
   sum_{j in R(Y_i)} Z_nj exp(-X_j' b_n).

2. (6.40) prints the second term as (1 + 1/t) log[1 + t dA(Y_i)], with
   the jump dA rather than the level A.  (6.39), directly above it, has
   [1 + theta Lambda_0(y)]^{1+1/theta} with the level.  The level is
   used here.

3. (6.42) is written for the uncensored case the section discusses.
   The posterior mean of a gamma frailty given d events is
   (1 + theta d) / (1 + theta Lambda_0(Y)), which reduces to the printed
   formula at d = 1; the d factor is carried here so that right-censored
   observations, which contribute no event, get the correct E step.

The likelihood maximised in step 2' is therefore

    log L(t, b, A) = sum_i { log dA(Y_i) - X_i' b
                             - (1 + 1/t) log[1 + t A(Y_i) exp(-X_i' b)] },

maximised over (log t, b) by cyclic coordinate golden-section search.
Nothing is random.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_ph_heterogeneity"]

_GR = 0.6180339887498949


def horowitz_ph_heterogeneity(t, x, event=None, frailty_dist="gamma",
                              theta=None, em_iter=60, cycles=8, gs_iter=44,
                              tol=1e-11):
    """Gamma-frailty proportional hazards: beta, the baseline, and theta.

    Parameters
    ----------
    t : array-like
        Observed durations Y.
    x : array-like
        n-by-p covariate matrix WITHOUT an intercept.
    event : array-like, optional
        1 for an event, 0 for right censoring; all ones if omitted.
    frailty_dist : str
        Only "gamma" is offered: it is the distribution p. 205 assumes.
    theta : float, optional
        Pin the frailty variance instead of estimating it.  theta -> 0 is
        the no-frailty limit, in which the M step is the Tsiatis baseline
        and beta is the partial-likelihood estimate.
    em_iter, cycles, gs_iter, tol :
        Deterministic optimiser controls.

    Returns
    -------
    beta_hat    : the coefficients, in the books exp(-x b) convention
    h0_hat      : the jumps of Lambda_n0 at the event times
    sigma2_hat  : theta, the variance of the gamma frailty
    """
    tt = core.vec(t)
    XX = core.mat(x)
    n = len(tt)
    if n == 0:
        raise ValueError("horowitz_ph_heterogeneity: t is empty")
    if len(XX) != n:
        raise ValueError("horowitz_ph_heterogeneity: x has a different number of rows than t")
    if frailty_dist != "gamma":
        raise ValueError("horowitz_ph_heterogeneity: only the gamma frailty of p.205 is offered")
    p = len(XX[0])
    if event is None:
        ev = [1.0] * n
    else:
        ev = core.vec(event)
        if len(ev) != n:
            raise ValueError("horowitz_ph_heterogeneity: event has a different length than t")
    order = sorted(range(n), key=lambda i: (tt[i], i))
    evt = [i for i in order if ev[i] != 0.0]
    if len(evt) == 0:
        raise ValueError("horowitz_ph_heterogeneity: no events")

    def mstep(Z, b):
        """(6.41) with the corrected denominator: jumps of Lambda_n0."""
        jumps = []
        for i in evt:
            s = 0.0
            for j in range(n):
                if tt[j] < tt[i]:
                    continue
                e = 0.0
                for k in range(p):
                    e += XX[j][k] * b[k]
                s += Z[j] * math.exp(min(max(-e, -300.0), 300.0))
            jumps.append(1.0 / s if s > 0.0 else 0.0)
        return jumps

    def cumulative(jumps):
        A = [0.0] * n
        run = 0.0
        pos = 0
        for i in order:
            if ev[i] != 0.0:
                run += jumps[pos]
                pos += 1
            A[i] = run
        return A

    def negll(par, A, dA):
        tau = par[0]
        b = par[1:]
        th = math.exp(min(max(tau, -30.0), 30.0))
        tot = 0.0
        for idx in range(len(evt)):
            i = evt[idx]
            if dA[idx] > 0.0:
                tot += math.log(dA[idx])
        for i in range(n):
            e = 0.0
            for k in range(p):
                e += XX[i][k] * b[k]
            w = math.exp(min(max(-e, -300.0), 300.0))
            if ev[i] != 0.0:
                tot -= e
            tot -= (1.0 + 1.0 / th) * math.log(1.0 + th * A[i] * w)
        return -tot

    b = [0.0] * p
    th = 1.0 if theta is None else float(theta)
    if th <= 0.0:
        raise ValueError("horowitz_ph_heterogeneity: theta must be positive")
    Z = [1.0] * n
    jumps = mstep(Z, b)
    A = cumulative(jumps)
    par = [math.log(th)] + list(b)
    for _ in range(int(em_iter)):
        prev = list(par)
        # step 2': maximise over (log theta, b) with A held fixed
        cur = negll(par, A, jumps)
        for _ in range(int(cycles)):
            for c in range(len(par)):
                if c == 0 and theta is not None:
                    continue
                lo = par[c] - 1.5
                hi = par[c] + 1.5
                a1 = hi - _GR * (hi - lo)
                a2 = lo + _GR * (hi - lo)
                q = list(par)
                q[c] = a1
                f1 = negll(q, A, jumps)
                q[c] = a2
                f2 = negll(q, A, jumps)
                for _ in range(int(gs_iter)):
                    if f1 < f2:
                        hi = a2
                        a2 = a1
                        f2 = f1
                        a1 = hi - _GR * (hi - lo)
                        q[c] = a1
                        f1 = negll(q, A, jumps)
                    else:
                        lo = a1
                        a1 = a2
                        f1 = f2
                        a2 = lo + _GR * (hi - lo)
                        q[c] = a2
                        f2 = negll(q, A, jumps)
                newv = 0.5 * (lo + hi)
                q[c] = newv
                fv = negll(q, A, jumps)
                if fv < cur:
                    par[c] = newv
                    cur = fv
        th = math.exp(min(max(par[0], -30.0), 30.0))
        b = list(par[1:])
        # step 3' (E): (6.42) with Lambda_n0 exp(-X b) in place of Lambda_n0
        for i in range(n):
            e = 0.0
            for k in range(p):
                e += XX[i][k] * b[k]
            w = math.exp(min(max(-e, -300.0), 300.0))
            Z[i] = (1.0 + th * (1.0 if ev[i] != 0.0 else 0.0)) / (1.0 + th * A[i] * w)
        # step 4' (M)
        jumps = mstep(Z, b)
        A = cumulative(jumps)
        d = 0.0
        for c in range(len(par)):
            if abs(par[c] - prev[c]) > d:
                d = abs(par[c] - prev[c])
        if d < tol:
            break
    return RichResult(
        title="Proportional hazards with a gamma frailty",
        summary_lines=[("n", n), ("events", len(evt)), ("theta", th)],
        payload={
            "estimate": th,
            "sigma2_hat": th,
            "theta_hat": th,
            "beta_hat": b,
            "h0_hat": jumps,
            "Lambda0": [A[i] for i in evt],
            "event_times": [tt[i] for i in evt],
            "frailty": Z,
            "n": n,
            "n_events": len(evt),
            "method": "Horowitz (2009) Sec. 6.2.2 (6.40)-(6.42) EM, steps 1-4' p.208 with covariates",
        },
    )


def cheatsheet():
    return "hrzphv: Proportional hazards model with unobserved heterogeneity (frailty)"
