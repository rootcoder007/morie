# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Odds ratio from a 2x2 table."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_odds_ratio"]


def wasserman_odds_ratio(table):
    """
    Odds ratio of a 2x2 contingency table.

    Formula: OR = (n11 n00) / (n10 n01), with the table laid out as
    [[n11, n10], [n01, n00]] — row = exposure, column = outcome,
    cell 11 in the top-left. The Woolf log-scale standard error
    se(log OR) = sqrt(sum 1/n_ij) and its 95 percent CI ship in the
    payload. Any zero cell makes OR degenerate — refused with the
    standard advice (Haldane 0.5 correction is a caller decision,
    not a silent default).

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Counts [[n11, n10], [n01, n00]], all > 0.

    Returns
    -------
    result : dict
        Keys: estimate (OR), log_or, se (of log OR), ci_lower,
        ci_upper, n, method.

    References
    ----------
    Wasserman (2004), Ch 16 (case-control); Woolf (1955).

    Examples
    --------
    >>> out = wasserman_odds_ratio([[30, 10], [15, 45]])
    >>> out["estimate"]
    9.0
    >>> round(out["se"], 12) == round((1/30 + 1/10 + 1/15 + 1/45) ** 0.5, 12)
    True
    >>> out["ci_lower"] < 9.0 < out["ci_upper"]
    True
    >>> wasserman_odds_ratio([[5, 0], [3, 2]])
    Traceback (most recent call last):
        ...
    ValueError: a zero cell makes the odds ratio degenerate; apply a continuity correction explicitly if intended.
    """
    T = np.asarray(table, dtype=float)
    if T.shape != (2, 2):
        raise ValueError(f"the table must be 2x2; got shape {T.shape}.")
    if np.any(T < 0):
        raise ValueError("counts cannot be negative.")
    if np.any(T == 0):
        raise ValueError("a zero cell makes the odds ratio degenerate; "
                         "apply a continuity correction explicitly if intended.")
    n11, n10 = T[0]
    n01, n00 = T[1]
    or_ = (n11 * n00) / (n10 * n01)
    log_or = math.log(or_)
    se = math.sqrt(1/n11 + 1/n10 + 1/n01 + 1/n00)
    z = 1.959963984540054
    return RichResult(payload={
        "estimate": float(or_), "log_or": float(log_or), "se": float(se),
        "ci_lower": float(math.exp(log_or - z * se)),
        "ci_upper": float(math.exp(log_or + z * se)),
        "n": float(np.sum(T)),
        "method": "OR = n11 n00 / (n10 n01), Woolf log-scale CI"})


def cheatsheet():
    return "wsmodd: OR = ad/bc; se(log OR) = sqrt(sum 1/n); zero cell refused"
