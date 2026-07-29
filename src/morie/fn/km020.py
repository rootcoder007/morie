# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.20: the composite self-supervised loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_ssl_loss"]


def kamath_ch2_ssl_loss(L_PTi, lambda_i=None):
    """L_SSL = sum_i lambda_i L_PT_i; lambdas default to 1 (the book
    writes the unweighted sum with explicit coefficients).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.20, printed
    p. 50.

    Examples
    --------
    >>> kamath_ch2_ssl_loss([1.0, 2.0], [0.5, 0.25])["estimate"]
    1.0
    """
    L = np.atleast_1d(np.asarray(L_PTi, dtype=float))
    if len(L) == 0:
        raise ValueError("no pretext losses supplied.")
    lam = (np.ones(len(L)) if lambda_i is None
           else np.atleast_1d(np.asarray(lambda_i, dtype=float)))
    if len(lam) != len(L):
        raise ValueError("need one lambda per pretext loss.")
    if np.any(lam < 0):
        raise ValueError("negative task weights invert a loss into a "
                         "reward; refused.")
    return RichResult(payload={
        "estimate": float(np.dot(lam, L)),
        "components": [float(v) for v in lam * L],
        "lambdas": [float(v) for v in lam], "n": len(L),
        "method": "Composite SSL loss (Kamath Eq 2.20)"})


def cheatsheet():
    return "km020: weighted sum of pretext-task losses"
