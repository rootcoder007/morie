# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian mixture model density."""

from . import _array_core as np

from ._richresult import RichResult
from .wsmemt import wasserman_em_algorithm

__all__ = ["wasserman_gmm_em"]


def wasserman_gmm_em(X, k):
    """
    Two-component univariate GMM fit by EM.

    Formula: p(x) = sum_j pi_j N(x | mu_j, sigma_j^2). This module
    delegates the optimisation to wsmemt's EM (single source of
    truth) after building a deterministic initialisation from the
    data: components start at the lower/upper type-1 quartiles with
    the pooled sd and equal weights. Only k = 2 is implemented —
    a larger k raises rather than silently fitting something else.

    Parameters
    ----------
    X : array-like
        Sample, n >= 4.
    k : int
        Number of components; must be 2.

    Returns
    -------
    result : dict
        Keys: estimate (log-likelihood), weights, means, sds,
        iterations, converged, n, k, method.

    References
    ----------
    Wasserman (2004), Ch 19 (mixtures).

    Examples
    --------
    >>> X = [0.0, 0.1, -0.1, 0.05, 10.0, 10.1, 9.9, 10.05]
    >>> out = wasserman_gmm_em(X, 2)
    >>> [round(w, 6) for w in out["weights"]]
    [0.5, 0.5]
    >>> sorted(round(m, 4) for m in out["means"])
    [0.0125, 10.0125]
    >>> out["converged"]
    True
    >>> wasserman_gmm_em(X, 3)
    Traceback (most recent call last):
        ...
    ValueError: only the 2-component mixture is implemented; got k=3.
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = X.size
    if int(k) != 2:
        raise ValueError(f"only the 2-component mixture is implemented; got k={int(k)}.")
    if n < 4:
        raise ValueError("a 2-component mixture needs at least 4 points.")
    xs = np.sort(X)
    q1 = xs[int(np.ceil(0.25 * n)) - 1]
    q3 = xs[int(np.ceil(0.75 * n)) - 1]
    s = float(np.std(X, ddof=1))
    if s == 0:
        raise ValueError("a constant sample cannot support a mixture fit.")
    core = wasserman_em_algorithm(X, (0.5, float(q1), float(q3), s, s))
    return RichResult(payload={
        "estimate": core["log_likelihood"],
        "weights": [1.0 - core["pi"], core["pi"]],
        "means": [core["mu1"], core["mu2"]],
        "sds": [core["sd1"], core["sd2"]],
        "iterations": core["iterations"], "converged": core["converged"],
        "n": int(n), "k": 2,
        "method": "GMM k=2 via wsmemt EM; quartile+pooled-sd deterministic init"})


def cheatsheet():
    return "wsmgmm: delegates to wsmemt; init (0.5, q1, q3, s, s); k=2 only"
