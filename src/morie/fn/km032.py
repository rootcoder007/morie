# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.32: the span-corruption seq2seq loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_seq2seq_loss"]


def kamath_ch2_seq2seq_loss(x, xhat, i, j):
    """L = -(1/l_s) sum_{s=i..j} log P(x_s | x_hat, x_i:s-1) with
    l_s = j - i + 1 the span length. ``x`` holds the model probability
    of the true token at every position of the ORIGINAL sequence;
    ``xhat`` is the corrupted input, recorded for the signature.
    Indices are 0-based and inclusive.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.32, printed
    p. 54.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch2_seq2seq_loss([0.9, 0.5, 0.25, 0.9], "corrupted",
    ...                               1, 2)
    >>> abs(out["estimate"] - (math.log(2) + math.log(4)) / 2) < 1e-12
    True
    """
    p = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    i = int(i); j = int(j)
    if not 0 <= i <= j < len(p):
        raise ValueError(
            f"the span [{i}, {j}] must lie inside the sequence of "
            f"length {len(p)} with i <= j.")
    seg = p[i:j + 1]
    with np.errstate(divide="ignore"):
        losses = -np.log(seg)
    return RichResult(payload={
        "estimate": float(np.mean(losses)), "span_length": j - i + 1,
        "per_position": [float(v) for v in losses], "n": len(p),
        "method": "Span seq2seq loss (Kamath Eq 2.32)"})


def cheatsheet():
    return "km032: -mean log P over the decoded span [i, j]"
