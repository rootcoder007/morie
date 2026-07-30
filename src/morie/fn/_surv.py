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

import numpy as np

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
    n, p = X.shape
    beta = np.zeros(p)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    order = np.argsort(t)
    ts, es, Xs, offs = t[order], e[order], X[order], off[order]
    utimes = np.unique(ts[es == 1])

    loglik = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = Xs @ beta + offs
        eta = np.clip(eta, -500, 500)
        w = np.exp(eta)
        ll = 0.0
        U = np.zeros(p)
        I = np.zeros((p, p))
        for ut in utimes:
            at_risk = ts >= ut
            died = at_risk & (ts == ut) & (es == 1)
            d = int(died.sum())
            if d == 0:
                continue
            wr = w[at_risk]
            Xr = Xs[at_risk]
            wd = w[died]
            Xd = Xs[died]
            S0r = wr.sum()
            S1r = wr @ Xr
            S2r = (wr[:, None] * Xr).T @ Xr
            S0d = wd.sum()
            S1d = wd @ Xd
            S2d = (wd[:, None] * Xd).T @ Xd
            ll += eta[died].sum()
            U += Xd.sum(axis=0)
            if ties == "breslow" or d == 1:
                ll -= d * np.log(S0r)
                mu = S1r / S0r
                U -= d * mu
                I += d * (S2r / S0r - np.outer(mu, mu))
            elif ties == "efron":
                for l in range(d):
                    f = l / d
                    S0 = S0r - f * S0d
                    S1 = S1r - f * S1d
                    S2 = S2r - f * S2d
                    ll -= np.log(S0)
                    mu = S1 / S0
                    U -= mu
                    I += S2 / S0 - np.outer(mu, mu)
            else:
                raise ValueError('ties must be "breslow" or "efron"')
        try:
            step = np.linalg.solve(I, U)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(I, U, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            loglik = ll
            converged = True
            break
        loglik = ll
    return beta, float(loglik), I, U, int(it), bool(converged)


def baseline_hazard(t, e, X, beta, offset=None):
    """Breslow baseline cumulative hazard at each distinct event time."""
    n = t.size
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    w = np.exp(np.clip(X @ beta + off, -500, 500))
    utimes = np.unique(t[e == 1])
    dH = np.empty(utimes.size)
    for i, ut in enumerate(utimes):
        at_risk = t >= ut
        d = int(((t == ut) & (e == 1)).sum())
        dH[i] = d / max(w[at_risk].sum(), 1e-300)
    return utimes, dH, np.cumsum(dH)


def km_estimate(t, e):
    """Kaplan-Meier survival at each distinct event time."""
    utimes = np.unique(t[e == 1])
    surv = np.empty(utimes.size)
    s = 1.0
    for i, ut in enumerate(utimes):
        n_risk = int((t >= ut).sum())
        d = int(((t == ut) & (e == 1)).sum())
        s *= 1.0 - d / max(n_risk, 1)
        surv[i] = s
    return utimes, surv
