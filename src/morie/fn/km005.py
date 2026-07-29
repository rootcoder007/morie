# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.5: the decoder's next-token distribution."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_decoder_token_distribution"]


def kamath_ch2_decoder_token_distribution(s_t_1, y_t_1, c, W=None):
    """P(y_t' | ...) = softmax over vocabulary scores built from
    (s, y, c). With W (vocab x 3d) the scores are W [s; y; c]; without
    it, the concatenated features are used as the scores directly.
    The result sums to 1, asserted in the tests.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.5, printed
    p. 31 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> out = kamath_ch2_decoder_token_distribution([0.0], [0.0], [0.0])
    >>> sum(out["distribution"])
    1.0
    """
    feats = np.concatenate([
        np.atleast_1d(np.asarray(s_t_1, dtype=float)),
        np.atleast_1d(np.asarray(y_t_1, dtype=float)),
        np.atleast_1d(np.asarray(c, dtype=float))])
    if W is not None:
        Wm = np.atleast_2d(np.asarray(W, dtype=float))
        if Wm.shape[1] != len(feats):
            raise ValueError(
                f"W has {Wm.shape[1]} columns but the concatenated "
                f"features have {len(feats)}.")
        scores = Wm @ feats
    else:
        scores = feats
    z = scores - scores.max()
    p = np.exp(z) / np.exp(z).sum()
    return RichResult(payload={
        "distribution": [float(v) for v in p],
        "predicted_token": int(np.argmax(p)),
        "estimate": float(p.max()), "n": len(p),
        "method": "Decoder token distribution softmax (Kamath Eq 2.5)"})


def cheatsheet():
    return "km005: softmax over W[s; y; c] vocabulary scores"
