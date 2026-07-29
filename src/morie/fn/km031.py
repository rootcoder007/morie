# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.31: sentence order prediction (SOP)."""

import numpy as np

from ._richresult import RichResult
from .km030 import kamath_ch2_nsp_loss

__all__ = ["kamath_ch2_sop_loss"]


def kamath_ch2_sop_loss(x, y, d):
    """L_SOP = -log P(d | x, y): the same binary form as Eq 2.30 with
    d meaning IN-ORDER (1) versus swapped (0). The form is shared, so
    the implementation is too; what differs is the task the caller's
    model was trained on, which no formula can check.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.31, printed
    p. 54.

    Examples
    --------
    >>> import math
    >>> abs(kamath_ch2_sop_loss(0.5, "s2", 0)["estimate"]
    ...     - math.log(2)) < 1e-12
    True
    """
    inner = kamath_ch2_nsp_loss(x, y, d)
    return RichResult(payload={
        "estimate": inner["estimate"], "p_in_order": inner["p_next"],
        "label": inner["label"], "n": 1,
        "method": "Sentence order prediction loss (Kamath Eq 2.31)"})


def cheatsheet():
    return "km031: SOP shares Eq 2.30's binary form, d = in-order"
