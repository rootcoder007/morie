# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.36: GPT's supervised objective."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_supervised_obj"]


def kamath_ch2_gpt_supervised_obj(C, x=None, y=None):
    """L2(C) = sum over (x, y) of log P(y | x). ``C`` holds the
    model's probability of the TRUE label per example.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.36, printed
    p. 70.

    Examples
    --------
    >>> import math
    >>> abs(kamath_ch2_gpt_supervised_obj([0.5, 1.0])["estimate"]
    ...     + math.log(2)) < 1e-12
    True
    """
    p = np.atleast_1d(np.asarray(C, dtype=float))
    if len(p) == 0:
        raise ValueError("no examples supplied.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    with np.errstate(divide="ignore"):
        logs = np.log(p)
    return RichResult(payload={
        "estimate": float(logs.sum()),
        "mean_log_likelihood": float(logs.mean()), "n": len(p),
        "method": "GPT supervised objective L2 (Kamath Eq 2.36)"})


def cheatsheet():
    return "km036: sum log P(true label) over the labelled set"
