# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.9: image-text matching with hard negatives."""

from ._richresult import RichResult
from .km136 import kamath_ch9_mml_vlm_loss

__all__ = ["kamath_ch9_itm_hard_negative"]


def kamath_ch9_itm_hard_negative(Pos, HardNeg):
    r"""L_ITM-hn = -sum_Pos log p(aligned) - sum_HardNeg log p(unaligned).

    Eq 9.9 differs from Eq 9.8 only in WHICH negatives are drawn (high
    TF-IDF similarity ones), not in the arithmetic, so the loss itself
    is ``morie.fn.km136``; this wrapper records that the negatives are
    hard.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.9, printed
    p. 387.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_itm_hard_negative([0.5, 0.5], [0.5])
    >>> abs(out["estimate"] - 3 * math.log(2)) < 1e-12
    True
    """
    r = kamath_ch9_mml_vlm_loss(Pos, HardNeg)
    return RichResult(payload={
        "estimate": r["estimate"], "positive_loss": r["positive_loss"],
        "negative_loss": r["negative_loss"],
        "n_positive": r["n_positive"],
        "n_hard_negative": r["n_negative"], "n": r["n"],
        "method": "ITM loss with hard negatives (Kamath Eq 9.9; the "
                  "Eq 9.8 core in km136)"})


def cheatsheet():
    return "km137: km136 with a hard-negative pair set"
