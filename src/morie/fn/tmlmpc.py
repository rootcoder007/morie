# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a multi-state cause-specific cumulative hazard contrast."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_multi_state_phc"]


def tmle_multi_state_phc(time, state, D, X):
    """Targeted contrast of the cause-specific cumulative hazard.

    In a multi-state model each transition has its own hazard and the
    competing transitions remove people from the risk set, so a marginal
    contrast has to be built transition by transition.  The target here
    is the counterfactual cumulative hazard of the FIRST transition
    label, ``Lambda_k^a(t0) = E_W[sum_{j <= t0} h_k(j | a, W)]``, and the
    reported estimate is ``Lambda_k^1(t0) - Lambda_k^0(t0)``.

    Writing the parameter as a mean over ``W`` of a sum of conditional
    hazards makes its efficient influence function immediate.  The score
    for ``h_k(j)`` is ``R(j) (N_k(j) - h_k(j))`` with ``R(j)`` the
    at-risk indicator, so the clever covariate is

        ``H_k(j) = I(A = a) / g_a(W) / P(R(j) = 1 | a, W)``

    and the at-risk probability is the product of surviving every
    transition and not being censored, both from pooled logistic
    hazards on the person-period expansion.  A single ``eps`` per arm is
    solved by fixed-iteration Newton on the pooled score.

    Parameters
    ----------
    time : array-like, shape (n,)
        Time of the observed transition (or of censoring).
    state : array-like, shape (n,)
        Label of the state entered at ``time``; 0 means censored.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Baseline covariates.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``lam1``, ``lam0``, ``eps1``, ``eps0``,
        ``t0``, ``n``.

    References
    ----------
    Rytgaard, H. C., Gerds, T. A. & van der Laan, M. J. (2022).
    Continuous-time targeted minimum loss-based estimation of
    intervention-specific mean outcomes.  Annals of Statistics 50(5).
    doi:10.1214/21-AOS2114.
    """
    tv = C.vec(time)
    sv = C.vec(state)
    Dv = C.vec(D)
    n = len(tv)
    if n == 0 or len(sv) != n or len(Dv) != n:
        raise ValueError("tmle_multi_state_phc: time, state and D must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_multi_state_phc: X must have one row per subject")
    causes = sorted(set(v for v in sv if v > 0.5))
    if not causes:
        raise ValueError("tmle_multi_state_phc: no observed transitions")
    grid = sorted(set(tv[i] for i in range(n) if sv[i] > 0.5))
    K = len(grid)
    t0 = grid[-1]
    target = causes[0]
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]

    idx = []
    rows = []
    for i in range(n):
        for k in range(K):
            if grid[k] <= tv[i]:
                idx.append((i, k))
                rows.append([1.0, grid[k], Dv[i]] + list(Xm[i]))
    hb = {}
    for c in causes:
        hb[c] = S.glmbin(rows, [1.0 if (abs(sv[i] - c) < 1e-9 and grid[k] == tv[i]) else 0.0
                                for (i, k) in idx])
    cb = S.glmbin(rows, [1.0 if (sv[i] < 0.5 and grid[k] == tv[i]) else 0.0 for (i, k) in idx])

    def hz(b, i, k, a):
        return S.clip(S.expit(C.dot([1.0, grid[k], a] + list(Xm[i]), b)), 1e-8, 1.0 - 1e-8)

    def atrisk(i, a):
        """P(still at risk at each grid point | A = a, W_i), just before it."""
        out = []
        p = 1.0
        for k in range(K):
            out.append(p)
            tot = sum(hz(hb[c], i, k, a) for c in causes)
            p *= (1.0 - S.clip(tot, 1e-8, 1.0 - 1e-8)) * (1.0 - hz(cb, i, k, a))
        return out

    R1 = [atrisk(i, 1.0) for i in range(n)]
    R0 = [atrisk(i, 0.0) for i in range(n)]

    def arm(a, Rp):
        ga = [g[i] if a > 0.5 else 1.0 - g[i] for i in range(n)]
        Hf = [[(1.0 if abs(Dv[i] - a) < 0.5 else 0.0) / ga[i] / max(Rp[i][k], 1e-8)
               for k in range(K)] for i in range(n)]
        ybin = [1.0 if (abs(sv[i] - target) < 1e-9 and grid[k] == tv[i]) else 0.0
                for (i, k) in idx]
        eps = 0.0
        for _ in range(30):
            score = 0.0
            info = 0.0
            for r in range(len(idx)):
                i, k = idx[r]
                hv = Hf[i][k]
                p = S.clip(S.expit(S.logit(hz(hb[target], i, k, Dv[i])) + eps * hv), 1e-12, 1 - 1e-12)
                score += hv * (ybin[r] - p)
                info += hv * hv * p * (1.0 - p)
            if info < 1e-14:
                break
            step = score / info
            eps += step
            if abs(step) < 1e-13:
                break
        lam = [0.0] * n
        for i in range(n):
            for k in range(K):
                h = hz(hb[target], i, k, a)
                lam[i] += S.clip(S.expit(S.logit(h) + eps * Hf[i][k]), 1e-12, 1 - 1e-12)
        psi = sum(lam) / n
        ic = [0.0] * n
        for r in range(len(idx)):
            i, k = idx[r]
            hv = Hf[i][k]
            p = S.clip(S.expit(S.logit(hz(hb[target], i, k, Dv[i])) + eps * hv), 1e-12, 1 - 1e-12)
            ic[i] += hv * (ybin[r] - p)
        for i in range(n):
            ic[i] += lam[i] - psi
        return psi, eps, ic

    lam1, eps1, ic1 = arm(1.0, R1)
    lam0, eps0, ic0 = arm(0.0, R0)
    est = lam1 - lam0
    ic = [ic1[i] - ic0[i] for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": est, "se": se, "lam1": lam1, "lam0": lam0,
        "eps1": eps1, "eps0": eps0, "t0": t0, "n": n,
        "method": "TMLE for a cause-specific cumulative hazard contrast"})


def cheatsheet():
    return "tmlmpc: TMLE for a multi-state cumulative hazard contrast."
