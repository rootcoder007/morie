# morie.fn -- function file (rootcoder007/morie)
"""Chernozhukov-Lee-Rosen intersection bounds."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["chernozhukov_rosen_bounds"]


def chernozhukov_rosen_bounds(y, X=None, instrument=None, alpha=0.05,
                              gamma=None, beta=0.1):
    """
    Chernozhukov-Lee-Rosen intersection bounds

    Formula: theta = inf_v m(v); precision-corrected critical values

    An upper bound valid for every cell v of the instrument is valid at
    their MINIMUM, but plugging in the sample minimum is biased downward
    because the noisiest cell wins.  The half-median-unbiased estimator
    keeps only the cells within beta-level precision of the minimum --
    the estimated contact set -- and takes the minimum over that set of
    m_v - k se_v.  With one cell it reduces to the ordinary one-sided
    interval, and with zero sampling noise to the plain minimum.

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
        Precision level of the contact set; None uses
        1 - 1/log(n_cells + 1).
    beta : float
        Not used directly; retained for the contact-set width.

    Returns
    -------
    result : dict
        Keys: estimate (bound), bound, naive_min, cells, means, ses,
        contact_set, k_alpha, n_cells, n.

    References
    ----------
    Chernozhukov, Lee & Rosen (2013), Intersection Bounds: Estimation
    and Inference, Econometrica 81(2):667-737.
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
        gamma = 1.0 - 1.0 / math.log(V + 1.0) if V > 1 else 0.9
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie strictly in (0, 1)")
    k_gamma = core.qnorm(gamma)
    naive = min(means)
    # contact set: cells whose bound is within 2 k_gamma se of the minimum
    thr = min(means[v] + 2.0 * k_gamma * ses[v] for v in range(V))
    contact = [v for v in range(V) if means[v] - 2.0 * k_gamma * ses[v] <= thr]
    if not contact:
        contact = [min(range(V), key=lambda v: means[v])]
    # one-sided critical value over the contact set, Bonferroni in |V_hat|
    k_alpha = core.qnorm(1.0 - alpha / len(contact))
    bound = min(means[v] - k_alpha * ses[v] for v in contact)
    return RichResult(payload={
        "estimate": bound,
        "bound": bound,
        "naive_min": naive,
        "cells": [float(sz) for sz in sizes],
        "means": means,
        "ses": ses,
        "contact_set": contact,
        "k_alpha": k_alpha,
        "n_cells": V,
        "n": n,
        "method": "Chernozhukov-Lee-Rosen intersection bounds",
    })


def cheatsheet():
    return "chrbnd: Chernozhukov-Lee-Rosen intersection bounds"


# compact alias per ledger/NAMING.md
chernozhukovrosenbounds = chernozhukov_rosen_bounds
