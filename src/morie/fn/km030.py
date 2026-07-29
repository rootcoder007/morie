# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.30/2.31: sentence-pair classification losses."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_nsp_loss"]


def kamath_ch2_nsp_loss(x, y, d):
    """L_NSP = -log P(d | x, y). ``x`` is the model's probability that
    y follows x; ``d`` the truth (1 next, 0 not); ``y`` is carried for
    the signature and recorded. The scored probability is x when d = 1
    and 1 - x when d = 0.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.30, printed
    p. 54.

    Examples
    --------
    >>> import math
    >>> abs(kamath_ch2_nsp_loss(0.5, "s2", 1)["estimate"]
    ...     - math.log(2)) < 1e-12
    True
    """
    p = float(x)
    if not 0 <= p <= 1:
        raise ValueError("the model probability must lie in [0, 1].")
    d = int(d)
    if d not in (0, 1):
        raise ValueError("d must be 0 or 1.")
    scored = p if d == 1 else 1.0 - p
    loss = float(-np.log(scored)) if scored > 0 else float("inf")
    return RichResult(payload={
        "estimate": loss, "p_next": p, "label": d, "n": 1,
        "method": "Next sentence prediction loss (Kamath Eq 2.30)"})


def cheatsheet():
    return "km030: -log P(d | x, y), binary next-sentence loss"
