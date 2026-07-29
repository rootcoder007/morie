# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.3: the Bradley-Terry reward-model objective."""

import numpy as np

from ._richresult import RichResult
from .km065 import kamath_ch5_reward_loss_pairwise

__all__ = ["kamath_ch5_rm_bradley_terry"]


def kamath_ch5_rm_bradley_terry(x, y_w, y_l, r_theta):
    """L(theta) = -E[log sigma(r(x,y_w) - r(x,y_l))].

    Eq 5.1 with the winner already named: y_w is preferred, so this
    DELEGATES to km065 with i = 0 throughout rather than restating the
    same loss. Same value, one implementation.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.3, printed
    p. 200.

    Examples
    --------
    >>> import math
    >>> r = lambda x, y: {"w": 1.0, "l": 0.0}[y]
    >>> out = kamath_ch5_rm_bradley_terry(["p"], ["w"], ["l"], r)
    >>> abs(out["estimate"] - math.log(1 + math.exp(-1))) < 1e-12
    True
    """
    inner = kamath_ch5_reward_loss_pairwise(
        r_theta, x, y_w, y_l, [0] * len(list(x)))
    return RichResult(payload={
        "estimate": inner["estimate"], "margins": inner["margins"],
        "per_pair": inner["per_pair"], "n": inner["n"],
        "method": "Bradley-Terry reward-model loss (Kamath Eq 5.3)"})


def cheatsheet():
    return "km067: Eq 5.1 with the winner fixed, delegated to km065"
