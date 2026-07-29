# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: unsmoothed n-gram MLE probability."""

from ._richresult import RichResult

__all__ = ["burkov_ngram_mle"]


def burkov_ngram_mle(counts_ngram, counts_prefix):
    """P_MLE = count(ngram) / count(prefix).

    A zero prefix count is refused: 0/0 is not a probability, and
    returning 0 there is precisely the silent failure smoothing exists
    to fix.

    References: Burkov LM (2025), Ch 2, MLE for n-grams.

    Examples
    --------
    >>> burkov_ngram_mle(3, 4)["estimate"]
    0.75
    """
    c = float(counts_ngram); p = float(counts_prefix)
    if c < 0 or p < 0:
        raise ValueError("counts must be non-negative.")
    if p == 0:
        raise ValueError(
            "the prefix was never observed, so the MLE conditional is "
            "undefined (0/0); use smoothing or backoff.")
    if c > p:
        raise ValueError(
            f"count(ngram) = {c} exceeds count(prefix) = {p}, which is "
            "impossible: every ngram occurrence contains its prefix.")
    return RichResult(payload={
        "estimate": c / p, "count_ngram": c, "count_prefix": p, "n": int(p),
        "method": "N-gram MLE count/prefix (Burkov Ch 2)"})


def cheatsheet():
    return "bkngr: n-gram MLE probability count/prefix (Burkov Ch 2)"
