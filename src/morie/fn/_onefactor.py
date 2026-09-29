# morie.fn -- private helper (rootcoder007/morie)
"""One-factor maximum-likelihood fit shared by the subscale AVE and
composite-reliability modules (sav_*, scr_*)."""

from __future__ import annotations

import math

from . import _frame_core as pd


def subscale_matrix(data, items, prefix):
    """Rows of the subscale items: DataFrame columns (complete rows only) or
    the array as given."""
    if items is None:
        items = [f"{prefix}{i}" for i in range(1, 6)]
    if isinstance(data, pd.DataFrame):
        cols = [[float(v) for v in data[c].tolist()] for c in items]
        rows = [list(r) for r in zip(*cols)]
        return [r for r in rows if not any(math.isnan(v) for v in r)]
    X = data.tolist() if hasattr(data, "tolist") else [list(r) for r in data]
    return [[float(v) for v in r] for r in X]


def correlation(rows):
    """Pearson correlation matrix of the columns of ``rows``."""
    n, p = len(rows), len(rows[0])
    if n < 3 or p < 2:
        raise ValueError("need at least 3 respondents and 2 items")
    mu = []
    for j in range(p):
        s = 0.0
        for r in rows:
            s += r[j]
        mu.append(s / n)
    C = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(a, p):
            s = 0.0
            for r in rows:
                s += (r[a] - mu[a]) * (r[b] - mu[b])
            C[a][b] = C[b][a] = s
    sd = [math.sqrt(C[j][j]) for j in range(p)]
    if any(v <= 0.0 for v in sd):
        raise ValueError("an item has zero variance")
    return [[C[a][b] / (sd[a] * sd[b]) for b in range(p)] for a in range(p)]


def one_factor_ml(R, tol=1e-12, max_iter=200000):
    """Standardised loadings of the one-factor model fitted to the
    correlation matrix ``R`` by maximum likelihood, via the EM algorithm of
    Rubin and Thayer (1982): with beta = lambda' Psi^-1 / (1 + lambda'
    Psi^-1 lambda), the M-step is lambda = R beta' / (1 - beta lambda +
    beta R beta') and Psi = diag(R - lambda beta R). Starts from Thurstone's
    highest-correlation communalities. Uniquenesses are bounded below by
    0.005, the lower bound ``stats::factanal`` uses, so a Heywood case stops
    at the boundary instead of crawling toward zero. Returns ``(lambda, psi,
    iterations, converged)``."""
    p = len(R)
    lam = [math.sqrt(min(0.95, max(0.05, max(abs(R[j][k]) for k in range(p) if k != j)))) for j in range(p)]
    psi = [1.0 - v * v for v in lam]
    it, conv = 0, False
    while it < max_iter:
        it += 1
        a = [lam[j] / psi[j] for j in range(p)]
        den = 1.0
        for j in range(p):
            den += a[j] * lam[j]
        beta = [v / den for v in a]
        Rb = []
        for j in range(p):
            s = 0.0
            for k in range(p):
                s += R[j][k] * beta[k]
            Rb.append(s)
        czz = 1.0
        for j in range(p):
            czz += -beta[j] * lam[j] + beta[j] * Rb[j]
        new = [v / czz for v in Rb]
        psi = [max(R[j][j] - new[j] * Rb[j], 0.005) for j in range(p)]
        delta = max(abs(new[j] - lam[j]) for j in range(p))
        lam = new
        if delta < tol:
            conv = True
            break
    return lam, psi, it, conv


def ave_and_cr(rows):
    """Loadings (absolute), uniquenesses, AVE = mean(lambda^2) and composite
    reliability (sum lambda)^2 / ((sum lambda)^2 + sum(1 - lambda^2))."""
    lam, psi, it, conv = one_factor_ml(correlation(rows))
    lam = [abs(v) for v in lam]
    p = len(lam)
    s2 = 0.0
    sl = 0.0
    se = 0.0
    for v in lam:
        s2 += v * v
        sl += v
        se += 1.0 - v * v
    return {
        "loadings": lam,
        "uniquenesses": psi,
        "ave": s2 / p,
        "cr": sl * sl / (sl * sl + se),
        "iterations": it,
        "converged": conv,
    }
