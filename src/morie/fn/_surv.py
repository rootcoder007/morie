# morie.fn -- shared helpers (rootcoder007/morie)
"""Shared Cox / survival machinery.

The partial likelihood is fitted once here, by Newton-Raphson on the score and
observed information, so every module in the family (tie corrections,
residuals, stratification, frailty, competing risks) works from the same
coefficients rather than re-deriving them slightly differently.

Data convention throughout: ``time`` is the observed follow-up, ``event`` is 1
for the event of interest and 0 for right-censoring. The risk set at time
:math:`t` is everyone with ``time >= t``.
"""

from __future__ import annotations

import math

from . import _array_core as np

__all__ = ["prepare", "cox_fit", "baseline_hazard", "km_estimate"]


def prepare(time, event, X=None):
    """Coerce and validate survival inputs."""
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    e = np.atleast_1d(np.asarray(event, dtype=float)).ravel()
    if t.size != e.size:
        raise ValueError(f"time has {t.size} entries but event has {e.size}")
    if t.size == 0:
        raise ValueError("time must be non-empty")
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    if not np.all((e == 0) | (e == 1)):
        raise ValueError("event must be 0 (censored) or 1 (event)")
    if X is None:
        return t, e, None
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != t.size:
        Xm = Xm.T
    if Xm.shape[0] != t.size:
        raise ValueError(f"X has {Xm.shape[0]} rows but time has {t.size}")
    return t, e, Xm


def cox_fit(t, e, X, ties="efron", max_iter=50, tol=1e-9, offset=None):
    r"""Newton-Raphson on the Cox partial likelihood.

    Returns ``(beta, loglik, information, score, n_iter, converged)``.

    ``ties="breslow"`` treats each tied event as if it occurred alone against
    the full risk set; ``"efron"`` averages the risk-set contribution over the
    tied events, which is materially more accurate when ties are common and is
    the default for that reason.
    """
    if ties not in ("breslow", "efron"):
        raise ValueError('ties must be "breslow" or "efron"')
    n, p = X.shape
    beta = np.zeros(p)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    # One pass from the latest time down: the risk set at time u is every
    # subject with t >= u, so its sums S0, S1, S2 grow as earlier times are
    # reached. Plain floats, O(n p^2) per Newton step.
    order = sorted(range(n), key=lambda i: float(t[i]), reverse=True)
    ts = [float(t[i]) for i in order]
    es = [float(e[i]) == 1.0 for i in order]
    Xs = [[float(v) for v in np.atleast_1d(X[i])] for i in order]
    offs = [float(off[i]) for i in order]
    R = range(p)

    loglik = -np.inf
    converged = False
    it = 0
    U = np.zeros(p)
    info = np.zeros((p, p))
    for it in range(1, max_iter + 1):  # noqa: B007 - the count is returned after the loop
        bl = [float(v) for v in beta]
        eta = [min(500.0, max(-500.0, sum(x[j] * bl[j] for j in R) + o)) for x, o in zip(Xs, offs)]
        w = [math.exp(v) for v in eta]
        ll = 0.0
        Ul = [0.0] * p
        info_l = [[0.0] * p for _ in R]
        S0r, S1r, S2r = 0.0, [0.0] * p, [[0.0] * p for _ in R]
        i = 0
        while i < n:
            ut = ts[i]
            S0d, S1d, S2d = 0.0, [0.0] * p, [[0.0] * p for _ in R]
            d = 0
            while i < n and ts[i] == ut:
                wi, xi = w[i], Xs[i]
                S0r += wi
                for j in R:
                    S1r[j] += wi * xi[j]
                    for k in R:
                        S2r[j][k] += wi * xi[j] * xi[k]
                if es[i]:
                    d += 1
                    S0d += wi
                    ll += eta[i]
                    for j in R:
                        S1d[j] += wi * xi[j]
                        Ul[j] += xi[j]
                        for k in R:
                            S2d[j][k] += wi * xi[j] * xi[k]
                i += 1
            if d == 0:
                continue
            steps = [0.0] if (ties == "breslow" or d == 1) else [k / d for k in range(d)]
            mult = d if len(steps) == 1 else 1
            for f in steps:
                S0 = S0r - f * S0d
                mu = [(S1r[j] - f * S1d[j]) / S0 for j in R]
                ll -= mult * math.log(S0)
                for j in R:
                    Ul[j] -= mult * mu[j]
                    for k in R:
                        info_l[j][k] += mult * ((S2r[j][k] - f * S2d[j][k]) / S0 - mu[j] * mu[k])
        U = np.array(Ul)
        info = np.array(info_l)
        try:
            step = np.linalg.solve(info, U)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, U, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            loglik = ll
            converged = True
            break
        loglik = ll
    return beta, float(loglik), info, U, int(it), bool(converged)


def baseline_hazard(t, e, X, beta, offset=None):
    """Breslow baseline cumulative hazard at each distinct event time."""
    n = t.size
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    w = [float(v) for v in np.exp(np.clip(X @ beta + off, -500, 500))]
    tl = [float(v) for v in t]
    el = [float(v) == 1.0 for v in e]
    # one pass from the latest time down: the risk-set sum at u covers every t >= u
    order = sorted(range(n), key=lambda i: tl[i], reverse=True)
    times, dHs = [], []
    S0, i = 0.0, 0
    while i < n:
        ut = tl[order[i]]
        d = 0
        while i < n and tl[order[i]] == ut:
            S0 += w[order[i]]
            d += el[order[i]]
            i += 1
        if d:
            times.append(ut)
            dHs.append(d / max(S0, 1e-300))
    times.reverse()
    dHs.reverse()
    dH = np.array(dHs) if dHs else np.empty(0)
    return (np.array(times) if times else np.empty(0)), dH, np.cumsum(dH)


def km_estimate(t, e):
    """Kaplan-Meier survival at each distinct event time."""
    tl = [float(v) for v in t]
    el = [float(v) == 1.0 for v in e]
    n = len(tl)
    order = sorted(range(n), key=lambda i: tl[i])
    times, surv = [], []
    s, i = 1.0, 0
    while i < n:  # ascending: the number at risk at u is everyone not yet passed
        ut = tl[order[i]]
        n_risk = n - i
        d = 0
        while i < n and tl[order[i]] == ut:
            d += el[order[i]]
            i += 1
        if d:
            s *= 1.0 - d / max(n_risk, 1)
            times.append(ut)
            surv.append(s)
    return (np.array(times) if times else np.empty(0)), (np.array(surv) if surv else np.empty(0))
