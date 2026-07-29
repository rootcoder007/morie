# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.8: BERTScore precision."""

from ._richresult import RichResult
from .km119 import kamath_ch8_bertscore_recall

__all__ = ["kamath_ch8_bertscore_precision"]


def kamath_ch8_bertscore_precision(x, xhat, normalize=False):
    r"""P_BERT = (1/|xhat|) sum_j max_i <xhat_j, x_i>.

    Eq 8.8 is Eq 8.7 with the two texts swapped -- the inner product
    is symmetric -- so this delegates to ``morie.fn.km119`` with the
    arguments exchanged rather than repeating the greedy match.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.8, printed
    p. 325.

    Examples
    --------
    >>> out = kamath_ch8_bertscore_precision([[1.0, 0.0], [0.0, 1.0]],
    ...                                      [[1.0, 0.0]])
    >>> out["estimate"]        # the one candidate token matches exactly
    1.0
    """
    r = kamath_ch8_bertscore_recall(xhat, x, normalize=normalize)
    return RichResult(payload={
        "estimate": r["estimate"], "per_token": r["per_token"],
        "greedy_match": r["greedy_match"], "n": r["n"],
        "method": "BERTScore precision (Kamath Eq 8.8; km119 with the "
                  "texts swapped)"})


def cheatsheet():
    return "km120: mean over candidate tokens of best reference match"
