# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cramer-Rao lower bound."""

from ._richresult import RichResult

__all__ = ["wasserman_cramer_rao"]


def wasserman_cramer_rao(theta, n, I):
    """
    Cramer-Rao lower bound for an unbiased estimator.

    Formula: Var(T) >= 1 / (n I(theta)) where I is the PER-OBSERVATION
    Fisher information. theta itself does not enter the bound's
    arithmetic (I is already evaluated at theta); it is carried in
    the payload for provenance.

    Parameters
    ----------
    theta : float
        Parameter value the information was evaluated at.
    n : int
        Sample size, >= 1.
    I : float
        Per-observation Fisher information I(theta), > 0.

    Returns
    -------
    result : dict
        Keys: estimate (the bound), se_bound (its sqrt), theta, n,
        information, method.

    References
    ----------
    Wasserman (2004), Ch 9, Theorem 9.23 (Cramer-Rao).

    Examples
    --------
    N(theta, sigma^2 = 4): I = 1/4; n = 25 -> bound = 4/25.

    >>> out = wasserman_cramer_rao(0.0, 25, 0.25)
    >>> out["estimate"]
    0.16
    >>> out["se_bound"]
    0.4
    >>> wasserman_cramer_rao(0.0, 25, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: the Cramer-Rao bound needs I(theta) > 0; got 0.0.
    >>> wasserman_cramer_rao(0.0, 0, 1.0)
    Traceback (most recent call last):
        ...
    ValueError: the Cramer-Rao bound needs n >= 1; got 0.
    """
    theta = float(theta)
    n = int(n)
    I = float(I)
    if n < 1:
        raise ValueError(f"the Cramer-Rao bound needs n >= 1; got {n}.")
    if I <= 0:
        raise ValueError(f"the Cramer-Rao bound needs I(theta) > 0; got {I}.")
    bound = 1.0 / (n * I)
    return RichResult(payload={
        "estimate": float(bound), "se_bound": float(bound ** 0.5),
        "theta": theta, "n": n, "information": I,
        "method": "Cramer-Rao Var(T) >= 1/(n I(theta))"})


def cheatsheet():
    return "wsmcrl: bound 1/(n I); I is per-observation information"
