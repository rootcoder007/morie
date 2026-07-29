# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.8: the visual-linguistic matching (MML) loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_mml_vlm_loss"]


def _check_probs(p, name):
    q = np.atleast_1d(np.asarray(p, dtype=float))
    if q.size == 0:
        raise ValueError(f"{name} is empty; a matching loss needs at "
                         "least one pair on each side.")
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"{name} holds probabilities and must lie in "
                         "[0, 1].")
    return q


def kamath_ch9_mml_vlm_loss(Pos, Neg):
    r"""L = -sum_Pos log p(aligned) - sum_Neg log p(unaligned).

    ``Pos`` holds p(aligned | x, y) for the positive image-sentence
    pairs and ``Neg`` holds p(unaligned | x', y') for the negatives --
    each side is the probability the model puts on the CORRECT label,
    so both enter with a plain -log. The sum (not the mean) is
    returned, as printed.

    Eq 9.9 (ITM with hard negatives) is the same expression with a
    harder negative set, so ``morie.fn.km137`` delegates here.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.8, printed
    p. 386.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_mml_vlm_loss([0.9], [0.8])
    >>> abs(out["estimate"] - (-math.log(0.9) - math.log(0.8))) < 1e-12
    True
    """
    pos = _check_probs(Pos, "Pos")
    neg = _check_probs(Neg, "Neg")
    with np.errstate(divide="ignore"):
        lp = -np.log(pos)
        ln = -np.log(neg)
    return RichResult(payload={
        "estimate": float(lp.sum() + ln.sum()),
        "positive_loss": float(lp.sum()),
        "negative_loss": float(ln.sum()),
        "n_positive": int(pos.size), "n_negative": int(neg.size),
        "n": int(pos.size + neg.size),
        "method": "visual-linguistic matching loss (Kamath Eq 9.8)"})


def cheatsheet():
    return "km136: -sum log p(correct label) over positive and negative pairs"
