# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the marginal hazard ratio when proportional hazards fails."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_hazard_ratio"]


def tmle_hazard_ratio(time, event, D, X):
    """Targeted marginal hazard ratio over the observed follow-up.

    Under non-proportional hazards there is no single Cox coefficient to
    report, so the target parameter is built from the two marginal
    survival curves instead: the counterfactual survivals at the last
    observed event time are targeted separately and the contrast is the
    ratio of the marginal cumulative hazards,
    ``Lambda_a(t0) = -log S_a(t0)``.  That quantity is defined whether or
    not the hazards are proportional and coincides with the Cox hazard
    ratio when they are.

    The nuisance step is a pooled logistic hazard on the person-period
    expansion, exactly as in Moore & van der Laan (2009): one row per
    subject per at-risk grid point, response the event indicator at that
    grid point, design ``[1, t, A, W]``.  The targeting step fluctuates
    the hazard on the logit scale along the survival clever covariate

        ``H_a(t) = -I(A = a) / g_a(W) * S_a(t0 | W) / S_a(t- | W)``

    where ``S_a(t- | W)`` is the survival just before the current grid
    point.  Solving the pooled score for a single ``eps`` by Newton
    (fixed iteration count, so both language arms take the same path)
    gives the targeted hazards, and ``psi_a = mean_i S*_a(t0 | W_i)``.

    Determinism: the time grid is the sorted set of distinct observed
    event times, IRLS runs a fixed number of iterations, and the
    fluctuation is a fixed-iteration Newton with no line search.

    Parameters
    ----------
    time : array-like, shape (n,)
        Observed follow-up time, ``min(T, C)``.
    event : array-like, shape (n,)
        1 if the event was observed, 0 if right censored.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Baseline covariates.

    Returns
    -------
    RichResult
        ``estimate`` (marginal hazard ratio ``Lambda_1(t0)/Lambda_0(t0)``),
        ``se``, ``s1``, ``s0``, ``eps``, ``t0``, ``n``.

    References
    ----------
    Moore, K. L. & van der Laan, M. J. (2009).  Covariate adjustment in
    randomized trials with binary outcomes: targeted maximum likelihood
    estimation.  Statistics in Medicine 28(1):39-64.
    doi:10.1002/sim.3445.  The targeting step is van der Laan, M. J. &
    Rubin, D. (2006), Targeted maximum likelihood learning,
    International Journal of Biostatistics 2(1):11.
    """
    tv = C.vec(time)
    ev = C.vec(event)
    Dv = C.vec(D)
    n = len(tv)
    if n == 0 or len(ev) != n or len(Dv) != n:
        raise ValueError("tmle_hazard_ratio: time, event and D must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_hazard_ratio: X must have one row per subject")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    grid = sorted(set(tv[i] for i in range(n) if ev[i] > 0.5))
    if not grid:
        raise ValueError("tmle_hazard_ratio: no observed events")
    K = len(grid)
    t0 = grid[-1]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]

    rows = []
    ybin = []
    idx = []
    for i in range(n):
        for k in range(K):
            if grid[k] <= tv[i]:
                rows.append([1.0, grid[k], Dv[i]] + list(Xm[i]))
                ybin.append(1.0 if (ev[i] > 0.5 and grid[k] == tv[i]) else 0.0)
                idx.append((i, k))
    hb = S.glmbin(rows, ybin)

    def haz(i, k, a):
        z = C.dot([1.0, grid[k], a] + list(Xm[i]), hb)
        return S.clip(S.expit(z), 1e-6, 1.0 - 1e-6)

    def curves(shift):
        """Survival curves per subject per arm under a logit shift."""
        out = []
        for i in range(n):
            arm = []
            for a in (0.0, 1.0):
                sv = []
                p = 1.0
                for k in range(K):
                    h = haz(i, k, a)
                    if shift is not None:
                        h = S.clip(S.expit(S.logit(h) + shift[i][int(a)][k]), 1e-12, 1.0 - 1e-12)
                    p *= 1.0 - h
                    sv.append(p)
                arm.append(sv)
            out.append(arm)
        return out

    S0 = curves(None)

    def clever(surv):
        """H_a(i, k) from the untargeted survival curves."""
        H = []
        for i in range(n):
            arm = []
            for a in (0.0, 1.0):
                ga = g[i] if a > 0.5 else 1.0 - g[i]
                hit = 1.0 if abs(Dv[i] - a) < 0.5 else 0.0
                row = []
                for k in range(K):
                    prev = surv[i][int(a)][k - 1] if k > 0 else 1.0
                    row.append(-hit / ga * surv[i][int(a)][K - 1] / prev)
                arm.append(row)
            H.append(arm)
        return H

    H = clever(S0)

    eps = 0.0
    for _ in range(30):
        score = 0.0
        info = 0.0
        for r in range(len(idx)):
            i, k = idx[r]
            a = int(round(Dv[i]))
            hv = H[i][a][k]
            p = S.clip(S.expit(S.logit(haz(i, k, Dv[i])) + eps * hv), 1e-12, 1.0 - 1e-12)
            score += hv * (ybin[r] - p)
            info += hv * hv * p * (1.0 - p)
        if info < 1e-14:
            break
        step = score / info
        eps += step
        if abs(step) < 1e-13:
            break

    shift = [[[eps * H[i][a][k] for k in range(K)] for a in (0, 1)] for i in range(n)]
    Sst = curves(shift)
    psi = [sum(Sst[i][a][K - 1] for i in range(n)) / n for a in (0, 1)]
    if psi[0] <= 0.0 or psi[0] >= 1.0 or psi[1] <= 0.0 or psi[1] >= 1.0:
        raise ValueError("tmle_hazard_ratio: targeted survival left (0, 1); grid too coarse")
    L0 = -math.log(psi[0])
    L1 = -math.log(psi[1])
    est = L1 / L0

    ic0 = [0.0] * n
    ic1 = [0.0] * n
    for r in range(len(idx)):
        i, k = idx[r]
        a = int(round(Dv[i]))
        hv = H[i][a][k]
        p = S.clip(S.expit(S.logit(haz(i, k, Dv[i])) + eps * hv), 1e-12, 1.0 - 1e-12)
        term = hv * (ybin[r] - p)
        if a == 1:
            ic1[i] += term
        else:
            ic0[i] += term
    for i in range(n):
        ic0[i] += Sst[i][0][K - 1] - psi[0]
        ic1[i] += Sst[i][1][K - 1] - psi[1]
    d1 = -1.0 / (psi[1] * L0)
    d0 = L1 / (psi[0] * L0 * L0)
    ic = [d1 * ic1[i] + d0 * ic0[i] for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": est, "se": se, "s1": psi[1], "s0": psi[0], "eps": eps,
        "t0": t0, "n": n,
        "method": "TMLE for the marginal hazard ratio under non-proportional hazards"})


def cheatsheet():
    return "tmlhrz: TMLE for the marginal hazard ratio under non-PH."
