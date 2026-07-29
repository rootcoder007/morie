# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markov inequality P(X >= a) <= E[X] / a."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_markov_ineq"]


def wasserman_markov_ineq(mean, a):
    """
    Markov inequality for a non-negative random variable.

    Formula: P(X >= a) <= E[X] / a, for X >= 0 and a > 0. The bound
    is reported capped at 1 (a probability), with the raw ratio
    alongside. A negative mean is refused: Markov's inequality
    applies to non-negative X, whose expectation cannot be negative.

    Parameters
    ----------
    mean : float
        E[X] of the non-negative variable (>= 0).
    a : float
        Threshold, strictly positive.

    Returns
    -------
    result : dict
        Keys: estimate (capped bound), raw_bound, mean, a, method.

    References
    ----------
    Wasserman (2004), Ch 4, Theorem 4.1.

    Examples
    --------
    >>> wasserman_markov_ineq(1.0, 4.0)["estimate"]
    0.25
    >>> wasserman_markov_ineq(3.0, 2.0)["estimate"]
    1.0
    >>> wasserman_markov_ineq(3.0, 2.0)["raw_bound"]
    1.5
    >>> wasserman_markov_ineq(-1.0, 2.0)
    Traceback (most recent call last):
        ...
    ValueError: Markov's inequality needs E[X] >= 0 (X non-negative); got -1.0.
    >>> wasserman_markov_ineq(1.0, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: Markov's inequality needs a > 0; got 0.0.
    """
    mean = float(mean)
    a = float(a)
    if mean < 0:
        raise ValueError(f"Markov's inequality needs E[X] >= 0 (X non-negative); got {mean}.")
    if a <= 0:
        raise ValueError(f"Markov's inequality needs a > 0; got {a}.")
    raw = mean / a
    return RichResult(payload={
        "estimate": float(min(raw, 1.0)), "raw_bound": float(raw),
        "mean": mean, "a": a,
        "method": "Markov bound E[X]/a (capped at 1)"})


def cheatsheet():
    return "wsmmrk: P(X >= a) <= E[X]/a for X >= 0, a > 0"
