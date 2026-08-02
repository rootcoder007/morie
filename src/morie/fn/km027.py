# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.27: the translation language modelling (TLM) loss."""

from . import _array_core as np

from ._richresult import RichResult
from .km022 import kamath_ch2_mlm_loss

__all__ = ["kamath_ch2_tlm_loss"]


def kamath_ch2_tlm_loss(x, y, M_x, M_y):
    """L_TLM = MLM over the source's masked positions + MLM over the
    target's -- each normalised by ITS OWN mask size, as the book's
    two -(1/|M|) sums state, then added. Both sides delegate to the
    Eq 2.22 implementation, so the three losses cannot drift apart.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.27, printed
    p. 53.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_tlm_loss([0.5], [0.25], [0], [0])
    >>> abs(out["estimate"] - (math.log(2) + math.log(4))) < 1e-12
    True
    """
    lx = kamath_ch2_mlm_loss(x, M_x)
    ly = kamath_ch2_mlm_loss(y, M_y)
    return RichResult(payload={
        "estimate": lx["estimate"] + ly["estimate"],
        "source_loss": lx["estimate"], "target_loss": ly["estimate"],
        "n": lx["n"] + ly["n"],
        "method": "TLM = source MLM + target MLM (Kamath Eq 2.27)"})


def cheatsheet():
    return "km027: source + target MLM, each over its own mask"
