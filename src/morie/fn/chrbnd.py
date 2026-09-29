# morie.fn -- function file (rootcoder007/morie)
"""Chernozhukov-Lee-Rosen intersection bounds."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["chernozhukov_rosen_bounds"]


def _kmax(p, m):
    """p-quantile of the maximum of m independent standard normals."""
    return core.qnorm(p ** (1.0 / m))


def chernozhukov_rosen_bounds(y, X=None, instrument=None, alpha=0.05, gamma=None, beta=0.1):
    """
    Chernozhukov-Lee-Rosen intersection bounds (independent cells).

    Target: theta0 = min_v theta(v), the intersection of the upper bounds
    theta(v) = E[Y | cell v]. The sample minimum of the cell means is biased
    DOWNWARD (the noisiest cell tends to win), so CLR correct each cell for
    its precision before taking the minimum:

        theta_hat(p) = min_{v in V_hat} [ m_v + k_{V_hat}(p) s_v ],

    where ``k_S(p)`` is the p-quantile of the maximum of the studentised
    estimation errors over S -- for independent cells, the p-quantile of
    the maximum of |S| independent standard normals, ``qnorm(p^(1/|S|))`` --
    and the contact set keeps only the cells that can be the minimiser,

        V_hat = { v : m_v <= min_u (m_u + k_V(gamma_n) s_u) + 2 k_V(gamma_n) s_v },

    with ``gamma_n = 1 - 0.1 / log(n)``. ``theta_hat(1 - alpha)`` is the
    one-sided (1 - alpha) upper confidence bound on theta0 and
    ``theta_hat(1/2)`` the half-median-unbiased estimate. (This module used
    to SUBTRACT a Bonferroni multiple of the standard error and to take the
    precision level from the number of cells, which with three cells made
    the contact-set constant negative.)

    Parameters
    ----------
    y : array-like
        Outcome.
    X : array-like or None
        Ignored; kept for the stub signature.
    instrument : array-like or None
        Cell label per observation; None puts all in one cell.
    alpha : float
        One-sided level of the reported bound.
    gamma : float or None
        Contact-set precision level; ``1 - 0.1 / log(n)`` when None.
    beta : float
        Unused; kept for signature stability.

    Returns
    -------
    result : dict
        Keys: estimate and bound (the 1 - alpha upper bound), hmu_estimate
        (half-median-unbiased), naive_min, cells, means, ses, contact_set,
        k_alpha, k_gamma, n_cells, n.

    References
    ----------
    Chernozhukov, Lee & Rosen (2013), Intersection Bounds: Estimation
    and Inference, Econometrica 81(2):667-737, Sec. 3 (precision
    correction, adaptive inequality selection with gamma_n = 1 - 0.1/log n).

    Examples
    --------
    >>> r = chernozhukov_rosen_bounds([3.0, 3.5, 2.8, 1.0, 1.6, 1.2, 5.0, 4.1],
    ...                               instrument=[0, 0, 0, 1, 1, 1, 2, 2])
    >>> r["contact_set"]
    [1]
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly in (0, 1)")
    ids = [0] * n if instrument is None else list(instrument)
    if len(ids) != n:
        raise ValueError("y and instrument must have the same length")
    keys = []
    for k in ids:
        if k not in keys:
            keys.append(k)
    means, ses, sizes = [], [], []
    for k in keys:
        vals = [yv[i] for i in range(n) if ids[i] == k]
        m = len(vals)
        if m < 2:
            raise ValueError("every instrument cell needs two observations")
        mu = sum(vals) / m
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (m - 1))
        means.append(mu)
        ses.append(sd / math.sqrt(m))
        sizes.append(m)
    V = len(keys)
    if gamma is None:
        gamma = 1.0 - 0.1 / math.log(n) if n > 1 else 0.9
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie strictly in (0, 1)")
    k_gamma = _kmax(gamma, V)
    thr = min(means[v] + k_gamma * ses[v] for v in range(V))
    contact = [v for v in range(V) if means[v] <= thr + 2.0 * k_gamma * ses[v]]
    k_alpha = _kmax(1.0 - alpha, len(contact))
    k_half = _kmax(0.5, len(contact))
    bound = min(means[v] + k_alpha * ses[v] for v in contact)
    hmu = min(means[v] + k_half * ses[v] for v in contact)
    return RichResult(
        payload={
            "estimate": bound,
            "bound": bound,
            "hmu_estimate": hmu,
            "naive_min": min(means),
            "cells": [float(sz) for sz in sizes],
            "means": means,
            "ses": ses,
            "contact_set": contact,
            "k_alpha": k_alpha,
            "k_gamma": k_gamma,
            "n_cells": V,
            "n": n,
            "method": "Chernozhukov-Lee-Rosen intersection bounds, independent cells",
        }
    )


def cheatsheet():
    return "chrbnd: Chernozhukov-Lee-Rosen intersection bounds"


# compact alias per ledger/NAMING.md
chernozhukovrosenbounds = chernozhukov_rosen_bounds
