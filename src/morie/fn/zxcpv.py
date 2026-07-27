# morie.fn -- function file (rootcoder007/morie)
"""Vine (C-vine) copula fitted to spatial/multivariate data."""

import numpy as np
from scipy import stats

from ._copula import copula_tau, tau_to_theta
from ._richresult import RichResult

__all__ = ["copula_vine_sp"]


def copula_vine_sp(data):
    r"""Fit a vine (c-vine) copula to multivariate data by inversion of Kendall's tau.

    Ranks each column to pseudo-observations, then fits the vine
    dependence structure by matching Kendall's tau pairwise (Czado
    2019 Table 3.2, p. 54).
    For the vine, a C-vine first tree is selected by taking the variable with the largest total absolute tau as the root, and each edge is fitted independently -- the greedy Dissmann selection restricted to tree 1.
    
    

    Parameters
    ----------
    data : array-like, shape (n, d)
        Observations, one column per variable.

    Returns
    -------
    RichResult
        keys: ``tau_matrix`` (d, d), ``pseudo_obs`` (n, d), ``theta_matrix``, ``root``, ``tree1_edges``, ``tree1_theta``, ``n``, ``d``, ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3 (bivariate families and tau), Table 3.2 p. 54; Ch. 5 (vine structure selection).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2-D (n observations x d variables).")
    n, d = X.shape
    if d < 2:
        raise ValueError("need at least 2 variables.")
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("data must be finite.")

    U = np.column_stack([stats.rankdata(X[:, j]) / (n + 1) for j in range(d)])
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            t = float(stats.kendalltau(X[:, i], X[:, j]).statistic)
            tau[i, j] = tau[j, i] = 0.0 if not np.isfinite(t) else t

    strength = np.abs(tau).sum(axis=1) - 1.0
    root = int(np.argmax(strength))
    edges = [(root, j) for j in range(d) if j != root]
    th = []
    for (a, b) in edges:
        t = tau[a, b]
        fam_ij = "gumbel" if t > 0 else "frank"
        try:
            th.append((fam_ij, float(tau_to_theta(fam_ij, t)) if abs(t) > 1e-6 else None))
        except ValueError:
            th.append((fam_ij, None))

    return RichResult(
        payload={
            "tau_matrix": tau,
            "pseudo_obs": U,
            "theta_matrix": None,
            "root": root,
            "tree1_edges": edges,
            "tree1_theta": th,
            "n": int(n),
            "d": int(d),
            "method": "C-vine tree 1: max-total-tau root, per-edge family and theta",
        }
    )


def cheatsheet():
    return "zxcpv: vine copula fitted by pairwise Kendall-tau inversion"
