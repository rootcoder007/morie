# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parametric bootstrap."""

import numpy as np

from ._richresult import RichResult
from .wsmnpb import _lcg_uniforms

__all__ = ["wasserman_parametric_boot"]


def wasserman_parametric_boot(data, f, T, B, seed=13):
    """
    Parametric bootstrap SE.

    Formula: draw X*_b ~ f(.; theta_hat), compute theta*_b = T(X*_b),
    se = sqrt((1/B) sum (theta*_b - theta*_bar)^2). The sampler ``f``
    receives (theta_hat, u) where u is an array of n LCG uniforms,
    and must return the simulated sample — inversion sampling keeps
    the whole run deterministic and language-portable. ``f = None``
    means the exponential model: X = -log(1-u) * theta_hat (mean
    theta_hat), whose MLE for the mean IS the sample mean.

    Parameters
    ----------
    data : array-like
        Observed sample (non-empty).
    f : callable or None
        Sampler f(theta_hat, u) -> array of len(u) draws.
    T : callable or None
        Statistic (None = sample mean).
    B : int
        Replications, >= 2.
    seed : int
        LCG seed (default 13).

    Returns
    -------
    result : dict
        Keys: estimate (theta_hat = T(data)), se, se_unbiased,
        replicates_mean, B, n, method.

    References
    ----------
    Wasserman (2004), Ch 8 (parametric bootstrap); Ch 9 for the MLE
    plug-in.

    Examples
    --------
    >>> out = wasserman_parametric_boot([1.0, 2.0, 3.0, 4.0], None, None, 300)
    >>> out["estimate"]
    2.5
    >>> 0.8 < out["se"] < 2.2   # exponential mean has se theta/sqrt(n) = 1.25
    True
    >>> out["B"]
    300
    >>> wasserman_parametric_boot([1.0], None, None, 1)
    Traceback (most recent call last):
        ...
    ValueError: the bootstrap needs B >= 2; got 1.
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    B = int(B)
    if n == 0:
        raise ValueError("the bootstrap of an empty sample is undefined.")
    if B < 2:
        raise ValueError(f"the bootstrap needs B >= 2; got {B}.")
    if T is None:
        T = lambda a: float(np.mean(a))
    if f is None:
        f = lambda theta, u: -np.log(1.0 - u) * theta
    theta_hat = float(T(data))
    u = _lcg_uniforms(B * n, seed).reshape(B, n)
    reps = np.array([float(T(np.asarray(f(theta_hat, u[b]), dtype=float)))
                     for b in range(B)])
    rbar = float(np.mean(reps))
    se_b = float(np.sqrt(np.mean((reps - rbar) ** 2)))
    se_u = float(np.sqrt(np.sum((reps - rbar) ** 2) / (B - 1)))
    return RichResult(payload={
        "estimate": theta_hat, "se": se_b, "se_unbiased": se_u,
        "replicates_mean": rbar, "B": B, "n": int(n),
        "method": "parametric bootstrap, inversion sampler f(theta,u), LCG"})


def cheatsheet():
    return "wsmprb: X* ~ f(.;theta_hat) via inversion on LCG uniforms; se over T(X*)"
