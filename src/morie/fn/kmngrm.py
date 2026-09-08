# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-gram language model conditional probability (MLE)."""

from ._richresult import RichResult
from .bkngr import burkov_ngram_mle

__all__ = ["kamath_ngram_language_model"]


def kamath_ngram_language_model(counts_ngram, counts_prefix):
    """P(w_t | w_{t-n+1..t-1}) = count(w_{t-n+1..t}) / count(prefix).

    Kamath's n-gram MLE is Burkov's n-gram MLE, so this DELEGATES to
    ``morie.fn.bkngr`` instead of keeping a second copy of one
    division that would eventually disagree with the first. The
    refusals come with it: an unseen prefix is 0/0 (undefined, not 0)
    and an n-gram count above its prefix count is impossible.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 1,
    n-gram language models.

    Examples
    --------
    >>> kamath_ngram_language_model(3, 4)["estimate"]
    0.75
    >>> kamath_ngram_language_model(0, 5)["estimate"]
    0.0
    """
    base = burkov_ngram_mle(counts_ngram, counts_prefix)
    return RichResult(payload={
        "estimate": float(base["estimate"]),
        "probability": float(base["estimate"]),
        "count_ngram": float(base["count_ngram"]),
        "count_prefix": float(base["count_prefix"]),
        "n": int(base["n"]),
        "method": "N-gram MLE conditional probability (delegates to bkngr)"})


def cheatsheet():
    return "kmngrm: count(ngram)/count(prefix) via bkngr; 0/0 refused"
