# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cluster-robust double machine learning for the ATE.

A general policy-evaluation entry point for double/debiased machine learning
when observations are clustered -- flights within an airspace corridor,
students within a school, patients within a hospital. Standard DML standard
errors assume independence and are anti-conservative under within-cluster
correlation. This cross-fits the AIPW (doubly-robust) score and computes a
cluster-robust variance from per-cluster score sums (Liang & Zeger 1986,
one-way; Cameron, Gelbach & Miller 2011, up to two-way).

Nuisances are cross-fitted (Chernozhukov et al. 2018): a logistic propensity
and per-arm OLS outcome regressions, so the AIPW point estimate is
Neyman-orthogonal. Only the SE is cluster-aware.

R parity: ``rmorie`` ``R/dml_clustered.R`` (``morie_dml_clustered``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _design(data: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Intercept + covariate columns as a float design matrix."""
    X = data[covariates].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def _ps(Xtr, dtr, Xte, eps):
    """Logistic propensity via IRLS; predicted, clipped to [eps, 1-eps]."""
    beta = np.zeros(Xtr.shape[1])
    for _ in range(25):
        eta = Xtr @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        # Ridge-stabilised IRLS step.
        WX = Xtr * w[:, None]
        H = Xtr.T @ WX + 1e-6 * np.eye(Xtr.shape[1])
        g = Xtr.T @ (dtr - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    beta[~np.isfinite(beta)] = 0.0
    eta = Xte @ beta
    return np.clip(1.0 / (1.0 + np.exp(-eta)), eps, 1 - eps)


def _ols(X, idx, y, te):
    """Per-arm OLS prediction; robust to thin arms and rank deficiency."""
    p = X.shape[1]
    if idx.size < p + 2:
        val = y[idx].mean() if idx.size else y.mean()
        return np.full(te.size, val)
    Xm = X[idx]
    try:
        beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ y[idx])
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Xm.T @ Xm) @ (Xm.T @ y[idx])
    return X[te] @ beta


def _cluster_se(infl: np.ndarray, cluster: np.ndarray, n: int) -> float:
    """Liang-Zeger one-way cluster-robust SE of a mean, from the IF."""
    total = 0.0
    for g in pd.unique(cluster):
        s = infl[cluster == g].sum()
        total += s * s
    return float(np.sqrt(max(total / (n * n), 0.0)))


def _multiway_se(infl, clusters, n):
    if len(clusters) == 1:
        return _cluster_se(infl, clusters[0], n)
    a, b = clusters[0], clusters[1]
    inter = np.array([f"{ai}|{bi}" for ai, bi in zip(a, b)])
    va = _cluster_se(infl, a, n) ** 2
    vb = _cluster_se(infl, b, n) ** 2
    vab = _cluster_se(infl, inter, n) ** 2
    return float(np.sqrt(max(va + vb - vab, 0.0)))


def dml_clustered(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    cluster: str | list[str] | None = None,
    n_folds: int = 5,
    seed: int = 123,
    eps: float = 0.02,
    ps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cluster-robust DML estimate of the ATE.

    Parameters mirror ``morie_dml_clustered``. ``cluster`` is a column name
    (one-way), a length-2 list (two-way), or None (i.i.d. SE). Returns a dict
    with ``ate``, ``se``, ``ci95``, ``z``, ``pval``, ``n``, ``n_clusters``,
    ``se_kind``.
    """
    cl_cols = [] if cluster is None else ([cluster] if isinstance(cluster, str) else list(cluster))
    if len(cl_cols) > 2:
        raise ValueError("`cluster` supports at most two-way clustering")
    keep = list(dict.fromkeys([treatment, outcome, *covariates, *cl_cols]))
    missing = [c for c in keep if c not in data.columns]
    if missing:
        raise ValueError("columns not found: " + ", ".join(missing))
    data = data.dropna(subset=keep).reset_index(drop=True)

    d = data[treatment].to_numpy()
    uy = np.unique(d)
    if not set(np.unique(d)).issubset({0, 1}):
        if uy.size != 2:
            raise ValueError("`treatment` must be binary")
        d = (d == uy[1]).astype(float)
    d = d.astype(float)
    y = data[outcome].to_numpy(dtype=float)
    X = _design(data, covariates)
    n = y.size

    rng = np.random.default_rng(seed)
    folds = rng.permutation(np.tile(np.arange(n_folds), n // n_folds + 1)[:n])
    e_hat = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    for k in range(n_folds):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        if te.size == 0:
            continue
        if ps is None:
            e_hat[te] = _ps(X[tr], d[tr], X[te], eps)
        for dv in (1.0, 0.0):
            idx = tr[d[tr] == dv]
            pred = _ols(X, idx, y, te)
            if dv == 1.0:
                mu1[te] = pred
            else:
                mu0[te] = pred
    if ps is not None:
        e_hat = np.clip(np.asarray(ps, dtype=float), eps, 1 - eps)

    psi = (mu1 - mu0) + d * (y - mu1) / e_hat - (1 - d) * (y - mu0) / (1 - e_hat)
    ate = float(psi.mean())
    infl = psi - ate

    if not cl_cols:
        se = float(psi.std(ddof=1) / np.sqrt(n))
        se_kind = "iid"
        n_clusters = None
    else:
        cls = [data[c].astype(str).to_numpy() for c in cl_cols]
        se = _multiway_se(infl, cls, n)
        se_kind = "cluster-robust (1-way)" if len(cl_cols) == 1 else "cluster-robust (2-way, CGM)"
        n_clusters = int(pd.unique(cls[0]).size)

    z = ate / se if se > 0 else 0.0
    # 2-sided normal p-value.
    from math import erf, sqrt

    pval = 2 * (0.5 * (1 - erf(abs(z) / sqrt(2))))
    return {
        "ate": ate,
        "se": se,
        "ci95": (ate - 1.96 * se, ate + 1.96 * se),
        "z": z,
        "pval": pval,
        "n": n,
        "n_clusters": n_clusters,
        "se_kind": se_kind,
    }
