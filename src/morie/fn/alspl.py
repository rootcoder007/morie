# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ancestral sampling from decoder logits (Alammar Ch 6)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_sampling_decoding"]


def alammar_sampling_decoding(logits, seed=0):
    """y_t ~ Categorical(softmax(logits_t)) per step, driven by the
    shared LCG so the R mirror draws the SAME tokens: u = (s + 0.5) /
    2^32, s = (1664525 s + 1013904223) mod 2^32, token = smallest v
    with cumulative probability > u.

    Examples
    --------
    >>> alammar_sampling_decoding([[9.0, 0.0], [0.0, 9.0]],
    ...                           seed=1)["tokens"]
    [0, 1]
    """
    Z = np.atleast_2d(np.asarray(logits, dtype=float))
    s = int(seed) % 2 ** 32
    toks = []
    us = []
    for row in Z:
        z = row - row.max()
        p = np.exp(z) / np.exp(z).sum()
        s = (1664525 * s + 1013904223) % 2 ** 32
        u = (s + 0.5) / 2 ** 32
        us.append(u)
        c = np.cumsum(p)
        toks.append(int(np.searchsorted(c, u, side="right")))
    return RichResult(payload={
        "tokens": toks, "uniforms": [float(u) for u in us],
        "estimate": float(toks[0]), "n": len(toks),
        "method": "Ancestral sampling via shared LCG (Alammar Ch 6)"})


def cheatsheet():
    return "alspl: softmax-categorical sampling on the exact-integer LCG"
