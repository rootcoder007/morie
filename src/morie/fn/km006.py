# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.6: seq2seq cross-entropy over the target sequence."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_seq2seq_cross_entropy"]


def kamath_ch2_seq2seq_cross_entropy(y, c, U=None):
    """L = -sum_t log p(y_t | y_<t, c).

    ``y`` holds the target token indices; ``c`` the model's per-step
    distributions, one row per target position (the conditional in the
    formula IS the model output here). U must equal len(y) when given.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.6, printed
    p. 31 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_seq2seq_cross_entropy([0], [[0.5, 0.5]])
    >>> round(out["estimate"], 10) == round(math.log(2), 10)
    True
    """
    idx = np.atleast_1d(np.asarray(y)).astype(int)
    P = np.atleast_2d(np.asarray(c, dtype=float))
    if P.shape[0] != len(idx):
        raise ValueError(
            f"need one distribution row per target token; got "
            f"{P.shape[0]} rows for {len(idx)} tokens.")
    if U is not None and int(U) != len(idx):
        raise ValueError(f"U = {U} does not match the sequence length "
                         f"{len(idx)}.")
    if np.any(np.abs(P.sum(axis=1) - 1.0) > 1e-8) or np.any(P < 0):
        raise ValueError("every row of c must be a probability "
                         "distribution.")
    if np.any((idx < 0) | (idx >= P.shape[1])):
        raise ValueError("a target index is outside the vocabulary.")
    picked = P[np.arange(len(idx)), idx]
    with np.errstate(divide="ignore"):
        losses = -np.log(picked)
    return RichResult(payload={
        "estimate": float(losses.sum()), "per_step": [float(v)
                                                      for v in losses],
        "mean_loss": float(losses.mean()), "n": len(idx),
        "method": "Seq2seq cross-entropy (Kamath Eq 2.6)"})


def cheatsheet():
    return "km006: -sum log p(y_t), distributions validated row by row"
