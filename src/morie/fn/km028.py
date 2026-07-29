# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.28: the alternate language modelling (ALM) pretraining loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_alm_loss"]


def _validate_probs(p, name):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if len(p) == 0:
        raise ValueError(f"{name} is empty.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"every entry of {name} must lie in [0, 1].")
    return p


def kamath_ch2_alm_loss(x, M):
    """L = -(1/|S|) sum over the masked positions of the code-switched sequence z of log P.

    ``x`` holds the model's probability of the TRUE token at every
    position; the index set selects which positions the loss reads.
    The sequence z(x, y) is the code-switched merge the book describes; the loss itself is the MLM form over it. A probability of 0 at a scored position gives infinite
    loss, which is the mathematics and is returned, not clipped.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.28, printed
    p. 53.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_alm_loss([0.5, 1.0, 0.25], [0, 2])
    >>> abs(out["estimate"] - (math.log(2) + math.log(4)) / 2) < 1e-12
    True
    """
    p = _validate_probs(x, "x")
    idx = np.atleast_1d(np.asarray(M)).astype(int)
    if len(idx) == 0:
        raise ValueError("the scored index set is empty; a loss over "
                         "nothing is not 0, it is undefined.")
    if np.any((idx < 0) | (idx >= len(p))):
        raise ValueError("an index lies outside the sequence.")
    if len(set(int(i) for i in idx)) != len(idx):
        raise ValueError("the index set contains duplicates.")
    with np.errstate(divide="ignore"):
        losses = -np.log(p[idx])
    return RichResult(payload={
        "estimate": float(np.mean(losses)),
        "per_position": [float(v) for v in losses],
        "positions_scored": [int(i) for i in idx], "n": len(p),
        "method": "alternate language modelling (ALM) loss (Kamath Eq 2.28)"})


def cheatsheet():
    return "km028: -mean log P over the masked positions of the code-switched sequence z"
