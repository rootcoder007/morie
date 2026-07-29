# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.2: BLEU modified (clipped) n-gram precision."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_bleu_precision"]


def kamath_ch8_bleu_precision(n_grams):
    r"""p_n = (# clipped matching n-grams) / (# n-grams generated).

    ``n_grams`` is one ``[clipped_matches, total_generated]`` row per
    n-gram order (a single pair is accepted for one order). Counting
    the clipped matches from raw text is ``kmbleu``'s job; this is the
    ratio itself.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.2, printed
    p. 323.

    Examples
    --------
    >>> out = kamath_ch8_bleu_precision([[3, 4], [1, 3]])
    >>> [round(v, 6) for v in out["p_n"]]
    [0.75, 0.333333]
    """
    A = np.atleast_2d(np.asarray(n_grams, dtype=float))
    if A.shape[-1] != 2:
        raise ValueError("give [clipped_matches, total_generated] per "
                         "n-gram order; got rows of width "
                         f"{A.shape[-1]}.")
    if np.any(A < 0):
        raise ValueError("n-gram counts cannot be negative.")
    if np.any(A[:, 1] == 0):
        raise ValueError("an n-gram order with zero generated n-grams "
                         "has an undefined precision (0/0); the "
                         "candidate is shorter than n.")
    if np.any(A[:, 0] > A[:, 1]):
        raise ValueError("clipped matches exceed the generated count; "
                         "clipping was not applied.")
    p = A[:, 0] / A[:, 1]
    est = float(p[0]) if p.size == 1 else [float(v) for v in p]
    return RichResult(payload={
        "estimate": est, "p_n": [float(v) for v in p],
        "n": int(p.size),
        "method": "BLEU clipped n-gram precision (Kamath Eq 8.2)"})


def cheatsheet():
    return "km114: clipped n-gram matches / generated n-grams"
