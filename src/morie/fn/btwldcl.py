# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Wild cluster bootstrap-t for clustered errors.

Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008),
"Bootstrap-Based Improvements for Inference with Clustered Errors",
*The Review of Economics and Statistics* 90(3), 414-427.  Read from the
NBER Technical Working Paper 344 text of the same paper; the two
load-bearing passages are section 3.2, which states the scheme as
"u*_g = u_g with probability 0.5 and u*_g = -u_g with probability 0.5,
with this assignment AT THE CLUSTER LEVEL", naming the +/-1 multipliers
Rademacher weights, and the CRVE finite-sample factor

    c = [G/(G-1)] [(N-1)/(N-k)],   with u_tilde_g = sqrt(c) u_hat_g.

So the whole cluster's residual vector is flipped by one shared sign,
which is what preserves the within-cluster correlation the CRVE is
there to handle.  Flipping observation by observation would silently
destroy it and would still produce plausible-looking numbers.

The Wald statistic is bootstrapped, not the coefficient (the paper's
"bootstrap-t"), because only the studentised version gets the asymptotic
refinement that makes this worth doing with few clusters:

    w*_b = (beta*_b - beta_hat) / se*_b,

with se*_b a cluster-robust standard error recomputed on each
pseudo-sample, and the two-sided p-value is the fraction of |w*_b|
at least |w| where w = (beta_hat - beta_0)/se.

Anchor: with G = N singleton clusters the cluster sign is an
observation-level Rademacher draw, so the bootstrap variance target
collapses to the HC0 sandwich; and in general Var*(beta*) is exactly the
UNCORRECTED clustered sandwich, (X'X)^{-1} (sum_g X_g' u_g u_g' X_g)
(X'X)^{-1}, because Var(v_g) = 1.  ``vcov_cluster0`` reports that target
directly and ``vcov_cluster`` applies the paper's c.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult
from .btres import _xtxinv

__all__ = ["boot_wild_cluster"]


def _crve(Xm, res, groups, n, p, corrected):
    """Clustered sandwich; ``corrected`` applies CGM's c = [G/(G-1)][(N-1)/(N-k)]."""
    XtXinv = _xtxinv(Xm, n, p)
    keys = []
    for gname in groups:
        if gname not in keys:
            keys.append(gname)
    G = len(keys)
    meat = [[0.0] * p for _ in range(p)]
    for gname in keys:
        sc = [0.0] * p
        for i in range(n):
            if groups[i] == gname:
                for j in range(p):
                    sc[j] += Xm[i][j] * res[i]
        for j in range(p):
            for k in range(p):
                meat[j][k] += sc[j] * sc[k]
    c = 1.0
    if corrected:
        if G < 2:
            raise ValueError("boot_wild_cluster: need at least two clusters")
        c = (G / (G - 1.0)) * ((n - 1.0) / (n - p))
    mid = [[sum(XtXinv[j][t] * meat[t][k] * c for t in range(p)) for k in range(p)] for j in range(p)]
    return [sum(mid[j][t] * XtXinv[t][j] for t in range(p)) for j in range(p)], keys, G


def boot_wild_cluster(X, y, cluster, B=200, seed=1, coef=1, beta0=0.0, alpha=0.05):
    """Wild cluster bootstrap-t for one coefficient.

    Parameters
    ----------
    X : array-like
        The n x p design.
    y : array-like
        The n responses.
    cluster : array-like
        Cluster label per observation.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    coef : int
        Zero-based index of the coefficient under test.
    beta0 : float
        Null value for that coefficient.
    alpha : float
        Test level; ``reject`` is the decision at this level.

    Returns
    -------
    RichResult
        ``beta_b``, ``w_b`` (bootstrap Wald statistics), ``beta_hat``,
        ``se_cluster``, ``w``, ``p_value``, ``reject``,
        ``vcov_cluster`` / ``vcov_cluster0`` (with and without CGM's c),
        ``G``, ``n``, ``p``, ``B``.
    """
    from . import _tail1core as C

    Xm = core.mat(X)
    yy = core.vec(y)
    n = core.nrow(Xm)
    p = core.ncol(Xm)
    groups = list(cluster)
    if n != len(yy) or n != len(groups):
        raise ValueError("boot_wild_cluster: X, y and cluster have different lengths")
    if n <= p:
        raise ValueError("boot_wild_cluster: need more rows than columns")
    if int(B) < 2:
        raise ValueError("boot_wild_cluster: need at least two replicates")
    j0 = int(coef)
    if not 0 <= j0 < p:
        raise ValueError("boot_wild_cluster: coef out of range")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_wild_cluster: alpha must lie strictly between 0 and 1")
    bh = core.lstsq(Xm, yy)
    fit = [sum(Xm[i][j] * bh[j] for j in range(p)) for i in range(n)]
    res = [yy[i] - fit[i] for i in range(n)]
    vc, keys, G = _crve(Xm, res, groups, n, p, True)
    vc0, _k, _G = _crve(Xm, res, groups, n, p, False)
    se = math.sqrt(vc[j0])
    w = (bh[j0] - float(beta0)) / se if se > 0.0 else float("nan")
    gidx = {k: t for t, k in enumerate(keys)}
    g = C.Lcg(seed)
    reps = []
    ws = []
    for _ in range(int(B)):
        v = [1.0 if g.unif() < 0.5 else -1.0 for _ in range(G)]
        ys = [fit[i] + res[i] * v[gidx[groups[i]]] for i in range(n)]
        bb = core.lstsq(Xm, ys)
        fb = [sum(Xm[i][j] * bb[j] for j in range(p)) for i in range(n)]
        rb = [ys[i] - fb[i] for i in range(n)]
        vb, _kk, _GG = _crve(Xm, rb, groups, n, p, True)
        sb = math.sqrt(vb[j0]) if vb[j0] > 0.0 else float("nan")
        reps.append(bb)
        ws.append((bb[j0] - bh[j0]) / sb if sb == sb and sb > 0.0 else float("nan"))
    good = [u for u in ws if u == u]
    cnt = sum(1 for u in good if abs(u) >= abs(w))
    pv = (cnt + 1.0) / (len(good) + 1.0) if good else float("nan")
    return RichResult(
        title="Wild cluster bootstrap-t (Cameron, Gelbach and Miller 2008)",
        summary_lines=[("G", G), ("n", n), ("w", w), ("p_value", pv)],
        payload={
            "beta_b": reps,
            "w_b": ws,
            "beta_hat": bh,
            "se_cluster": se,
            "w": w,
            "p_value": pv,
            "reject": 1.0 if (pv == pv and pv < a) else 0.0,
            "vcov_cluster": vc,
            "vcov_cluster0": vc0,
            "G": G,
            "n": n,
            "p": p,
            "B": int(B),
            "estimate": bh[j0],
            "method": "Cameron, Gelbach and Miller (2008) Rev. Econ. Statist. 90(3):414-427",
        },
    )


def cheatsheet():
    return "btwldcl: ONE Rademacher sign per cluster, not per observation; bootstrap the Wald stat, not beta"
