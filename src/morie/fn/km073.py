# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.9: the same preference, written as a sigmoid."""

import numpy as np

from ._richresult import RichResult
from .km072 import kamath_ch5_bradley_terry_pref

__all__ = ["kamath_ch5_pref_sigmoid_form"]


def _pair(r_star):
    if isinstance(r_star, dict):
        if "y_w" in r_star and "y_l" in r_star:
            return float(r_star["y_w"]), float(r_star["y_l"])
        vals = list(r_star.values())
    else:
        vals = list(r_star)
    if len(vals) != 2:
        raise ValueError(
            "r_star must hold exactly two rewards (winner, loser), or a "
            "mapping with the keys 'y_w' and 'y_l'; got "
            f"{len(vals)} values.")
    return float(vals[0]), float(vals[1])


def kamath_ch5_pref_sigmoid_form(r_star):
    """p*(y_w > y_l | x) = sigma(r*(x,y_w) - r*(x,y_l)).

    Algebraically identical to Eq 5.8 -- divide numerator and
    denominator by exp(r_w) -- so it DELEGATES to km072 rather than
    restating it. ``r_star`` is the (winner, loser) reward pair.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.9, printed
    p. 210.

    Examples
    --------
    >>> round(kamath_ch5_pref_sigmoid_form([1.0, 0.0])["estimate"], 10)
    0.7310585786
    >>> kamath_ch5_pref_sigmoid_form([0.0, 0.0])["estimate"]
    0.5
    """
    rw, rl = _pair(r_star)
    inner = kamath_ch5_bradley_terry_pref({"y_w": rw, "y_l": rl},
                                          "y_w", "y_l")
    return RichResult(payload={
        "estimate": inner["estimate"], "margin": inner["margin"],
        "r_w": rw, "r_l": rl, "n": 2,
        "method": "preference as sigmoid of the margin (Kamath Eq 5.9)"})


def cheatsheet():
    return "km073: sigma(r_w - r_l), Eq 5.8 rewritten"
