# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Delta method variance propagation."""

import math

from ._richresult import RichResult

__all__ = ["wasserman_delta_method"]


def wasserman_delta_method(theta_hat, se, g_prime):
    """
    Delta method: standard error of g(theta_hat).

    Formula: Var(g(theta_hat)) ~= g'(theta)^2 Var(theta_hat), so
    se(g) = |g'(theta_hat)| * se(theta_hat). A zero derivative is
    refused rather than returning se 0: the first-order delta method
    is degenerate there and a second-order expansion is needed.

    Parameters
    ----------
    theta_hat : float
        Point estimate.
    se : float
        Standard error of theta_hat, > 0.
    g_prime : float
        Derivative g'(theta) evaluated at theta_hat, nonzero.

    Returns
    -------
    result : dict
        Keys: estimate (se of g), variance, theta_hat, se_theta,
        g_prime, method.

    References
    ----------
    Wasserman (2004), Ch 5, Theorem 5.13.

    Examples
    --------
    g(theta) = theta^2 at theta_hat = 3, se = 0.5: g' = 6, se_g = 3.

    >>> out = wasserman_delta_method(3.0, 0.5, 6.0)
    >>> out["estimate"]
    3.0
    >>> out["variance"]
    9.0
    >>> wasserman_delta_method(3.0, 0.5, -6.0)["estimate"]
    3.0
    >>> wasserman_delta_method(1.0, 0.0, 2.0)
    Traceback (most recent call last):
        ...
    ValueError: the delta method needs se > 0; got 0.0.
    >>> wasserman_delta_method(1.0, 0.5, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: g'(theta) = 0: first-order delta method degenerate; use second order.
    """
    theta_hat = float(theta_hat)
    se = float(se)
    g_prime = float(g_prime)
    if se <= 0:
        raise ValueError(f"the delta method needs se > 0; got {se}.")
    if g_prime == 0:
        raise ValueError("g'(theta) = 0: first-order delta method degenerate; use second order.")
    se_g = abs(g_prime) * se
    return RichResult(payload={
        "estimate": float(se_g), "variance": float(se_g ** 2),
        "theta_hat": theta_hat, "se_theta": se, "g_prime": g_prime,
        "method": "delta method se(g) = |g'| se(theta)"})


def cheatsheet():
    return "wsmdlm: se(g(theta)) = |g'(theta)| se(theta); refuse g' = 0"
