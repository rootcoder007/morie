# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: linear interpolation of n-gram orders."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_ngram_interpolation"]


def burkov_ngram_interpolation(probs_by_order, lambdas):
    """P = sum_k lambda_k P_k with the lambdas summing to 1.

    Weights that do not sum to 1 are refused: the result would not be
    a probability and every downstream perplexity would be silently
    wrong.

    References: Burkov LM (2025), Ch 2, interpolation.

    Examples
    --------
    >>> round(burkov_ngram_interpolation([0.8, 0.2],
    ...                                [0.75, 0.25])["estimate"], 12)
    0.65
    """
    ps = np.atleast_1d(np.asarray(probs_by_order, dtype=float))
    ls = np.atleast_1d(np.asarray(lambdas, dtype=float))
    if ps.shape != ls.shape:
        raise ValueError(
            f"need one lambda per order; got {len(ps)} probabilities and "
            f"{len(ls)} lambdas.")
    if np.any(ls < 0) or abs(float(ls.sum()) - 1.0) > 1e-9:
        raise ValueError(
            f"lambdas must be non-negative and sum to 1; they sum to "
            f"{float(ls.sum()):.6g}.")
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    est = float(np.dot(ls, ps))
    return RichResult(payload={
        "estimate": est, "probs": [float(v) for v in ps],
        "lambdas": [float(v) for v in ls], "n": len(ps),
        "method": "Linear interpolation of n-gram orders (Burkov Ch 2)"})


def cheatsheet():
    return "bkintr: interpolated n-gram probability (Burkov Ch 2)"
