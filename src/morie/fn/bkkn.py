# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: Kneser-Ney smoothing."""

from ._richresult import RichResult

__all__ = ["burkov_kneser_ney"]


def burkov_kneser_ney(counts_ngram, counts_prefix, continuation_counts,
                      d=0.75):
    """P_KN = max(c - d, 0)/prefix + lambda * P_continuation.

    ``continuation_counts`` is ``(n_types_after_prefix,
    continuation_count_of_word, total_bigram_types)``: how many
    distinct words follow the prefix, in how many distinct contexts
    the word appears, and the number of distinct bigram types. The
    normaliser lambda = d * n_types_after_prefix / prefix makes the
    distribution over a full vocabulary sum to 1 -- asserted in the
    tests by summing it.

    References: Burkov LM (2025), Ch 2, Kneser-Ney.

    Examples
    --------
    >>> out = burkov_kneser_ney(2, 4, (2, 3, 10), d=0.75)
    >>> round(out["estimate"], 10)
    0.425
    """
    c = float(counts_ngram); p = float(counts_prefix); dd = float(d)
    if c < 0 or p <= 0:
        raise ValueError("need non-negative count and positive prefix.")
    if c > p:
        raise ValueError("count(ngram) cannot exceed count(prefix).")
    if not 0 < dd < 1:
        raise ValueError(f"the discount d must lie in (0, 1); got {d}.")
    n_after, cont_w, total_types = (float(v) for v in continuation_counts)
    if total_types <= 0 or cont_w < 0 or n_after < 0:
        raise ValueError("continuation counts must be non-negative with "
                         "positive total bigram types.")
    if cont_w > total_types:
        raise ValueError("a word cannot appear in more contexts than "
                         "there are bigram types.")
    lam = dd * n_after / p
    p_cont = cont_w / total_types
    est = max(c - dd, 0.0) / p + lam * p_cont
    return RichResult(payload={
        "estimate": est, "discounted_mle": max(c - dd, 0.0) / p,
        "lambda": lam, "p_continuation": p_cont, "n": int(p),
        "method": "Kneser-Ney smoothing (Burkov Ch 2)"})


def cheatsheet():
    return "bkkn: Kneser-Ney absolute discounting + continuation (Burkov Ch 2)"
