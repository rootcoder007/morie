# morie.fn -- function file (rootcoder007/morie)
"""Turnbull NPMLE for interval-censored survival."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["interval_censored_survival"]


def interval_censored_survival(L, R, event=None, n_iter=200):
    """Nonparametric survival when each event time is only bracketed.

    Kaplan-Meier needs to know where each event happened; interval
    censoring only says it happened somewhere in ``(L, R]``.  Turnbull
    insight is that the likelihood cannot distinguish points inside an
    interval at all, so all the mass sits on a finite set of
    ``[q_j, p_j]`` regions -- the innermost intervals -- and the NPMLE
    is a discrete distribution on those.  The self-consistency equation
    is an EM: split each observation mass over the regions it covers,
    then renormalise.

    Determinism: a fixed iteration count with no tolerance test, so two
    arms cannot stop on different sweeps.

    Formula: with ``alpha_ij = 1`` when region ``j`` lies inside
    observation ``i``, iterate
    ``p_j <- (1/n) sum_i alpha_ij p_j / sum_k alpha_ik p_k``.

    Parameters
    ----------
    L : array-like, shape (n,)
        Left endpoints; use 0 for left-censored observations.
    R : array-like, shape (n,)
        Right endpoints; use ``inf`` for right-censored observations.
    event : array-like, optional
        Kept for interface compatibility; an observation with
        ``R = inf`` is right-censored regardless.
    n_iter : int, default 200
        EM sweeps.

    Returns
    -------
    RichResult
        ``estimate`` (median survival time, the first region right
        endpoint where survival drops to 0.5 or below), ``p`` (mass per
        region), ``surv``, ``q``, ``r``, ``n``, ``m``.

    References
    ----------
    Turnbull, B. W. (1976).  The empirical distribution function with
    arbitrarily grouped, censored and truncated data.  Journal of the
    Royal Statistical Society B 38:290-295.  The self-consistency
    equation above is equation (10) of that paper.
    """
    Lv = C.vec(L)
    Rv = [float(t) for t in C.vec(R)]
    n = len(Lv)
    lefts = sorted(set(Lv))
    rights = sorted(set(t for t in Rv if t != float("inf")))
    regs = []
    for q in lefts:
        for r in rights:
            if q < r and not any(q < s < r for s in lefts + rights):
                regs.append((q, r))
    regs = sorted(set(regs))
    m = len(regs)
    if m == 0:
        return RichResult(payload={
            "estimate": float("nan"), "p": [], "surv": [], "q": [], "r": [],
            "n": n, "m": 0, "method": "Turnbull NPMLE, interval censoring"})
    alpha = [[1.0 if (Lv[i] <= regs[j][0] and regs[j][1] <= Rv[i]) else 0.0
              for j in range(m)] for i in range(n)]
    p = [1.0 / m] * m
    for _ in range(int(n_iter)):
        new = [0.0] * m
        for i in range(n):
            den = sum(alpha[i][j] * p[j] for j in range(m))
            if den <= 0.0:
                continue
            for j in range(m):
                if alpha[i][j]:
                    new[j] += p[j] / den
        tot = sum(new)
        p = [t / tot for t in new] if tot > 0.0 else p
    surv, acc = [], 0.0
    for j in range(m):
        acc += p[j]
        surv.append(1.0 - acc)
    med = float("nan")
    for j in range(m):
        if surv[j] <= 0.5:
            med = regs[j][1]
            break
    return RichResult(payload={
        "estimate": med, "p": p, "surv": surv,
        "q": [g[0] for g in regs], "r": [g[1] for g in regs], "n": n, "m": m,
        "method": "Turnbull NPMLE, interval censoring"})


def cheatsheet():
    return "ssintc: Turnbull NPMLE for interval-censored survival."
