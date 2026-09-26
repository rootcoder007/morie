# morie.fn -- function file (rootcoder007/morie)
"""Joinpoint survival regression."""

from __future__ import annotations

import math
from itertools import combinations

from . import _array_core as np

__all__ = ["jnpnt"]


def _jk(b, L):
    """J_k = int_0^L s^k e^{b s} ds for k = 0, 1, 2, stable as b -> 0."""
    x = b * L
    if abs(x) < 1e-3:
        out = []
        for k in range(3):
            acc, term = 0.0, 1.0
            for m in range(12):
                acc += term * L ** (m + k + 1) / (m + k + 1)
                term *= b / (m + 1)
            out.append(acc)
        return out
    e = math.exp(x)
    j0 = math.expm1(x) / b
    j1 = (L * e - j0) / b
    j2 = (L * L * e - 2.0 * j1) / b
    return [j0, j1, j2]


def _fit_joinpoint(time, event, taus, n_iter=100):
    """Continuous piecewise log-linear hazard,
    log h(u) = a + b u + sum_k d_k (u - tau_k)_+, by Newton-Raphson on
    the exact concave log-likelihood sum_events log h(t_i) - sum_i H(t_i).
    """
    taus = sorted(float(t) for t in taus)
    p = 2 + len(taus)

    def x_at(u):
        return [1.0, u] + [max(u - t, 0.0) for t in taus]

    ev = [t for t, e in zip(time, event) if e > 0]
    sx = [0.0] * p
    for t in ev:
        xv = x_at(t)
        for j in range(p):
            sx[j] += xv[j]
    d = len(ev)
    theta = [math.log(d / sum(time))] + [0.0] * (p - 1)

    def loglik_grad_hess(th, need=True):
        ll = sum(sx[j] * th[j] for j in range(p))
        g = list(sx)
        H = [[0.0] * p for _ in range(p)]
        for t in time:
            cuts = [0.0] + [tk for tk in taus if tk < t] + [t]
            for q in range(len(cuts) - 1):
                lo, hi = cuts[q], cuts[q + 1]
                if hi <= lo:
                    continue
                # on [lo, hi] the design is alpha + beta u
                alpha = [1.0, 0.0] + [(-tk if lo >= tk else 0.0) for tk in taus]
                beta = [0.0, 1.0] + [(1.0 if lo >= tk else 0.0) for tk in taus]
                c = sum(alpha[j] * th[j] for j in range(p))
                bb = sum(beta[j] * th[j] for j in range(p))
                E = math.exp(c + bb * lo)
                j0, j1, j2 = _jk(bb, hi - lo)
                i0 = E * j0
                ll -= i0
                if not need:
                    continue
                i1 = E * (lo * j0 + j1)
                i2 = E * (lo * lo * j0 + 2.0 * lo * j1 + j2)
                for j in range(p):
                    g[j] -= alpha[j] * i0 + beta[j] * i1
                    for m in range(p):
                        H[j][m] -= (alpha[j] * alpha[m] * i0
                                    + (alpha[j] * beta[m] + beta[j] * alpha[m]) * i1
                                    + beta[j] * beta[m] * i2)
        return ll, g, H

    ll, g, H = loglik_grad_hess(theta)
    for _ in range(n_iter):
        step = [float(v) for v in np.linalg.solve(np.asarray(H), np.asarray([-v for v in g]))]
        dec = sum(g[j] * step[j] for j in range(p))
        t_ = 1.0
        while True:
            cand = [theta[j] + t_ * step[j] for j in range(p)]
            try:
                llc = loglik_grad_hess(cand, need=False)[0]
            except OverflowError:
                llc = -math.inf
            if llc >= ll + 1e-4 * t_ * dec or t_ < 1e-12:
                break
            t_ *= 0.5
        theta = cand
        ll, g, H = loglik_grad_hess(theta)
        if dec < 1e-13:
            break
    return theta, ll


def jnpnt(
    time: np.ndarray,
    event: np.ndarray,
    *,
    max_joinpoints: int = 3,
) -> dict:
    """Joinpoint regression for survival hazard trends.

    The log-hazard is continuous and piecewise linear in time,
    log h(t) = a + b t + sum_k d_k (t - tau_k)_+ (the joinpoint model of
    Kim, Fay, Feuer and Midthune 2000, applied to the hazard), fitted by
    exact maximum likelihood: the cumulative hazard of each piece is the
    closed-form integral of exp(c + b u).  Joinpoints are searched over a
    grid of event-time percentiles and the number of joinpoints is chosen
    by BIC, -2 log L + (2 + 2k) log n (each joinpoint adds its location
    and its slope change).

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    max_joinpoints : int
        Maximum number of joinpoints to test.

    Returns
    -------
    dict
        n_joinpoints, joinpoints, slopes and intercepts per segment
        (log h = intercept + slope t on each), bic, log_likelihood,
        n_obs, n_events.
    """
    time = [float(v) for v in np.asarray(time, dtype=float).tolist()]
    event = [float(v) for v in np.asarray(event, dtype=float).tolist()]
    if len(time) != len(event):
        raise ValueError("time and event must have the same length")
    if any(t < 0 for t in time):
        raise ValueError("times must be non-negative")
    n = len(time)
    event_times = sorted(t for t, e in zip(time, event) if e > 0)
    d = len(event_times)

    if d < 4:
        return {
            "n_joinpoints": 0,
            "joinpoints": np.array([]),
            "slopes": np.array([0.0]),
            "intercepts": np.array([math.log(d / sum(time) + 1e-300)]),
            "bic": math.inf,
            "log_likelihood": 0.0,
            "n_obs": n,
            "n_events": int(d),
        }

    def segments(theta, taus):
        a, b = theta[0], theta[1]
        ints, slopes = [a], [b]
        for k, t in enumerate(taus):
            a -= theta[2 + k] * t
            b += theta[2 + k]
            ints.append(a)
            slopes.append(b)
        return ints, slopes

    ev_sorted = event_times
    m = min(10, d)
    candidates = [float(v) for v in np.percentile(ev_sorted, np.linspace(10, 90, m))]
    candidates = sorted(set(candidates))
    best = None
    for njp in range(max_joinpoints + 1):
        combos = [()] if njp == 0 else combinations(candidates, njp)
        for jp in combos:
            theta, ll = _fit_joinpoint(time, event, jp)
            npar = 2 + 2 * njp
            bic = -2.0 * ll + npar * math.log(n)
            if best is None or bic < best[4]:
                best = (njp, list(jp), theta, ll, bic)
    njp, jp, theta, ll, bic = best
    ints, slopes = segments(theta, jp)
    return {
        "n_joinpoints": njp,
        "joinpoints": np.array(jp),
        "slopes": np.array(slopes),
        "intercepts": np.array(ints),
        "bic": float(bic),
        "log_likelihood": float(ll),
        "n_obs": n,
        "n_events": int(d),
    }


jnpnt_fn = jnpnt


def cheatsheet() -> str:
    return "jnpnt(time, event) -> Joinpoint survival regression."
