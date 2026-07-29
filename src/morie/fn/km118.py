# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.6: ROUGE-N recall against reference summaries."""

from collections import Counter

from ._richresult import RichResult

__all__ = ["kamath_ch8_rouge_n"]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n])
                   for i in range(len(tokens) - n + 1))


def kamath_ch8_rouge_n(S, gram_n, candidate=None):
    r"""ROUGE-N = sum_S sum_gram Count_match / sum_S sum_gram Count.

    ``S`` is the set of reference summaries (each a token sequence),
    ``gram_n`` the n-gram order, and ``candidate`` the generated token
    sequence. Count_match is clipped at the reference count, per the
    book's "maximum number of times an n-gram occurs in BOTH".

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.6, printed
    p. 324.

    Examples
    --------
    >>> out = kamath_ch8_rouge_n([["the", "cat", "sat"]], 1,
    ...                          candidate=["the", "cat"])
    >>> round(out["estimate"], 6)        # 2 of the 3 reference unigrams
    0.666667
    """
    n = int(gram_n)
    if n < 1:
        raise ValueError(f"the n-gram order must be >= 1; got {n}.")
    if candidate is None:
        raise ValueError("candidate= (the generated token sequence) is "
                         "required; ROUGE compares two texts.")
    refs = list(S)
    if len(refs) == 0:
        raise ValueError("no reference summaries were given.")
    if refs and isinstance(refs[0], str):
        raise ValueError("S must be a list of token SEQUENCES, not a "
                         "list of strings; tokenize first.")
    cand = _ngrams(list(candidate), n)
    matched = total = 0
    for ref in refs:
        rc = _ngrams(list(ref), n)
        total += sum(rc.values())
        matched += sum(min(c, cand[g]) for g, c in rc.items())
    if total == 0:
        raise ValueError(
            f"the references contain no {n}-grams (they are shorter "
            "than n); ROUGE-N is 0/0 there.")
    return RichResult(payload={
        "estimate": matched / total, "matched": int(matched),
        "total_reference_ngrams": int(total), "gram_n": n,
        "n": len(refs),
        "method": f"ROUGE-{n} recall (Kamath Eq 8.6)"})


def cheatsheet():
    return "km118: clipped reference n-gram overlap / reference n-grams"
