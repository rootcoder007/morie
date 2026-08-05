# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Two-stage joint model for a longitudinal outcome and survival.

Henderson, Diggle and Dobson (2000), "Joint modelling of longitudinal
measurements and event time data", Biostatistics 1(4):465-480,
doi:10.1093/biostatistics/1.4.465, link a linear mixed model for the
repeated measurements to a proportional-hazards model for the event
time through a shared latent process.  Rizopoulos (2012), *Joint Models
for Longitudinal and Time-to-Event Data*, Chapman & Hall/CRC,
doi:10.1201/b12208, chapter 4, sets out the two-stage version
implemented here:

  stage 1  y_ij = x_ij' beta + z_ij' b_i + e_ij,  b_i ~ N(0, D),
           e_ij ~ N(0, sigma^2), fitted by EM;
  stage 2  lambda_i(t) = lambda_0(t) exp(gamma' W_i + eta b_i),
           a Cox model with the empirical Bayes b_i as covariate.

eta is the association parameter: eta = 0 means the longitudinal
trajectory carries no information about the hazard.  Only a random
intercept is used, so D is scalar; the Cox model is fitted by
Newton-Raphson on the Breslow partial likelihood.  Both stages are
deterministic, with a fixed iteration count, so the two language arms
land on the same numbers.

The two-stage estimator is biased towards zero relative to the full
joint likelihood -- that is a documented property of the method
(Rizopoulos 2012 s.4.1), not an artefact here.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["joint_longitudinal_survival"]


def _lmm_ri(y, X, grp, order, iters=200):
    """Random-intercept LMM by EM.  Returns beta, tau2 (D), sigma2, b."""
    n = len(y)
    p = len(X[0])
    g = len(order)
    idx = [[] for _ in range(g)]
    for i in range(n):
        idx[order.index(grp[i])].append(i)
    beta = core.lstsq(X, y)
    r0 = [y[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    s2 = sum(v * v for v in r0) / max(1, n - p)
    t2 = s2
    b = [0.0] * g
    for _ in range(iters):
        # E step
        eb = []
        vb = []
        for k in range(g):
            m = len(idx[k])
            prec = m / s2 + 1.0 / t2
            resid = sum(y[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in idx[k])
            eb.append((resid / s2) / prec)
            vb.append(1.0 / prec)
        # M step
        adj = [y[i] - eb[order.index(grp[i])] for i in range(n)]
        beta = core.lstsq(X, adj)
        ss = 0.0
        for k in range(g):
            for i in idx[k]:
                e = y[i] - sum(X[i][j] * beta[j] for j in range(p)) - eb[k]
                ss += e * e + vb[k]
        s2 = ss / n
        t2 = sum(eb[k] * eb[k] + vb[k] for k in range(g)) / g
        b = eb
    return beta, t2, s2, b


def _cox_breslow(t, d, Z, iters=50):
    """Cox PH by Newton-Raphson on the Breslow partial likelihood."""
    g = len(t)
    p = len(Z[0])
    beta = [0.0] * p
    order = sorted(range(g), key=lambda i: (t[i], i))
    for _ in range(iters):
        u = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        ll = 0.0
        for k in range(g):
            i = order[k]
            if d[i] != 1.0:
                continue
            risk = [order[m] for m in range(g) if t[order[m]] >= t[i]]
            s0 = 0.0
            s1 = [0.0] * p
            s2 = [[0.0] * p for _ in range(p)]
            for j in risk:
                e = math.exp(sum(Z[j][a] * beta[a] for a in range(p)))
                s0 += e
                for a in range(p):
                    s1[a] += e * Z[j][a]
                    for c in range(p):
                        s2[a][c] += e * Z[j][a] * Z[j][c]
            ll += sum(Z[i][a] * beta[a] for a in range(p)) - math.log(s0)
            for a in range(p):
                u[a] += Z[i][a] - s1[a] / s0
                for c in range(p):
                    H[a][c] += s2[a][c] / s0 - (s1[a] / s0) * (s1[c] / s0)
        step = core.ridgesolve(H, u, 1e-10)
        mx = 0.0
        for a in range(p):
            beta[a] += step[a]
            if abs(step[a]) > mx:
                mx = abs(step[a])
        if mx < 1e-12:
            break
    inv = [core.ridgesolve(H, [1.0 if a == j else 0.0 for a in range(p)], 1e-10) for j in range(p)]
    se = [math.sqrt(inv[j][j]) if inv[j][j] > 0 else float("nan") for j in range(p)]
    return beta, se, ll


def joint_longitudinal_survival(long_y, time, event, X, Z, cluster):
    """Two-stage joint model: random-intercept LMM then Cox with shared b_i.

    Parameters
    ----------
    long_y : array-like
        Longitudinal measurements, one per (subject, occasion) record.
    time : array-like
        Event or censoring time, one per record (constant within subject).
    event : array-like
        Event indicator, one per record (constant within subject).
    X : array-like or None
        Fixed-effect design for the longitudinal model; an intercept is
        prepended.
    Z : array-like or None
        Subject-level covariates for the survival model; may be None.
    cluster : array-like
        Subject label of each record.
    """
    y = core.vec(long_y)
    n = len(y)
    if n == 0:
        raise ValueError("joint_longitudinal_survival: long_y is empty")
    tv = core.vec(time)
    ev = core.vec(event)
    cl = core.vec(cluster)
    if len(tv) != n or len(ev) != n or len(cl) != n:
        raise ValueError("joint_longitudinal_survival: long_y, time, event and cluster have different lengths")
    for v in ev:
        if v not in (0.0, 1.0):
            raise ValueError("joint_longitudinal_survival: event must be 0 or 1")
    Xd = core.design(X, n)
    if len(Xd) != n:
        raise ValueError("joint_longitudinal_survival: X and long_y have different lengths")
    order = []
    for v in cl:
        if v not in order:
            order.append(v)
    order.sort()
    g = len(order)
    if g < 3:
        raise ValueError("joint_longitudinal_survival: need at least three subjects")
    st = [0.0] * g
    sd = [0.0] * g
    for i in range(n):
        k = order.index(cl[i])
        st[k] = tv[i]
        sd[k] = ev[i]
    if sum(sd) == 0:
        raise ValueError("joint_longitudinal_survival: no events observed")
    beta, t2, s2, b = _lmm_ri(y, Xd, cl, order)
    if Z is None:
        Zs = [[b[k]] for k in range(g)]
        names = 1
    else:
        rows = core.mat(Z)
        if len(rows) == g:
            Zs = [list(rows[k]) + [b[k]] for k in range(g)]
        elif len(rows) == n:
            first = {}
            for i in range(n):
                k = order.index(cl[i])
                if k not in first:
                    first[k] = list(rows[i])
            Zs = [first[k] + [b[k]] for k in range(g)]
        else:
            raise ValueError("joint_longitudinal_survival: Z must have one row per subject or per record")
        names = len(Zs[0])
    gam, gse, pll = _cox_breslow(st, sd, Zs)
    eta = gam[-1]
    return RichResult(
        title="Joint longitudinal-survival model (two stage)",
        summary_lines=[("subjects", g), ("association eta", eta), ("se", gse[-1])],
        payload={
            "estimate": eta,
            "eta": eta,
            "se": gse[-1],
            "gamma": gam,
            "gamma_se": gse,
            "beta": beta,
            "tau2": t2,
            "sigma2": s2,
            "icc": t2 / (t2 + s2),
            "b": b,
            "partial_loglik": pll,
            "n_subjects": float(g),
            "n_events": sum(sd),
            "n_covariates": float(names),
            "n": n,
            "method": "stage 1 random-intercept LMM by EM, stage 2 Cox on the empirical Bayes b_i, Henderson et al (2000)",
        },
    )


def cheatsheet():
    return "jntlmm: Joint longitudinal-survival model"


# compact alias per ledger/NAMING.md
jointlongitudinalsurvival = joint_longitudinal_survival
