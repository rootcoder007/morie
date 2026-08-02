# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Relative risk from a 2x2 table."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_relative_risk"]


def wasserman_relative_risk(table):
    """
    Relative risk of a 2x2 table.

    Formula: RR = p1 / p0 with p1 = n11/(n11+n10) the outcome rate
    among exposed (row 1) and p0 = n01/(n01+n00) among unexposed
    (row 2); layout [[n11, n10], [n01, n00]] as in wsmodd. The
    log-scale se = sqrt((1-p1)/n11 + (1-p0)/n01) (Katz) and 95
    percent CI ship alongside. Zero event counts are refused
    explicitly.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Counts [[n11, n10], [n01, n00]]; both event cells > 0 and
        both row totals > 0.

    Returns
    -------
    result : dict
        Keys: estimate (RR), risk_exposed, risk_unexposed, log_rr,
        se, ci_lower, ci_upper, n, method.

    References
    ----------
    Wasserman (2004), Ch 16; Katz et al (1978).

    Examples
    --------
    >>> out = wasserman_relative_risk([[30, 70], [10, 90]])
    >>> round(out["estimate"], 12)
    3.0
    >>> out["risk_exposed"]
    0.3
    >>> out["risk_unexposed"]
    0.1
    >>> out["ci_lower"] < 3.0 < out["ci_upper"]
    True
    >>> wasserman_relative_risk([[0, 5], [2, 3]])
    Traceback (most recent call last):
        ...
    ValueError: zero event counts make the relative risk degenerate.
    """
    T = np.asarray(table, dtype=float)
    if T.shape != (2, 2):
        raise ValueError(f"the table must be 2x2; got shape {T.shape}.")
    if np.any(T < 0):
        raise ValueError("counts cannot be negative.")
    n11, n10 = T[0]
    n01, n00 = T[1]
    r1, r0 = n11 + n10, n01 + n00
    if r1 == 0 or r0 == 0:
        raise ValueError("both exposure rows need at least one subject.")
    if n11 == 0 or n01 == 0:
        raise ValueError("zero event counts make the relative risk degenerate.")
    p1, p0 = n11 / r1, n01 / r0
    rr = p1 / p0
    log_rr = math.log(rr)
    se = math.sqrt((1 - p1) / n11 + (1 - p0) / n01)
    z = 1.959963984540054
    return RichResult(payload={
        "estimate": float(rr), "risk_exposed": float(p1),
        "risk_unexposed": float(p0), "log_rr": float(log_rr),
        "se": float(se),
        "ci_lower": float(math.exp(log_rr - z * se)),
        "ci_upper": float(math.exp(log_rr + z * se)),
        "n": float(np.sum(T)),
        "method": "RR = p_exposed/p_unexposed, Katz log-scale CI"})


def cheatsheet():
    return "wsmrrr: RR = (n11/r1)/(n01/r0); Katz se on log scale"
