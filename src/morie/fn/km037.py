# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.37: the combined fine-tuning objective."""

from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_combined_obj"]


def kamath_ch2_gpt_combined_obj(L_1, L_2, lam=0.5):
    """L3 = L2(C) + lambda L1(U): the auxiliary LM objective keeps the
    representation from collapsing onto the task.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.37, printed
    p. 70.

    Examples
    --------
    >>> kamath_ch2_gpt_combined_obj(-2.0, -1.0, 0.5)["estimate"]
    -2.0
    """
    l1 = float(L_1); l2 = float(L_2); l = float(lam)
    if l < 0:
        raise ValueError("lambda must be non-negative; a negative weight "
                         "turns the auxiliary objective into a penalty "
                         "on likelihood.")
    return RichResult(payload={
        "estimate": l2 + l * l1, "L1": l1, "L2": l2, "lambda": l, "n": 2,
        "method": "Combined objective L2 + lambda L1 (Kamath Eq 2.37)"})


def cheatsheet():
    return "km037: L2 + lambda L1"
