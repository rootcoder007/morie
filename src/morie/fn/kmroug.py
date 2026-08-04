# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ROUGE-N recall of a hypothesis against a reference."""

from collections import Counter

from ._richresult import RichResult

__all__ = ["kamath_rouge_n"]


def _toks(x):
    return x.split() if isinstance(x, str) else [str(t) for t in x]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def kamath_rouge_n(hypothesis, reference, n=1):
    """ROUGE-N = sum_g count_match(g) / sum_g count(g), over the
    reference's n-grams.

    Matches are CLIPPED at the reference count, so repeating a word
    five times cannot score it five times -- without that clip, "the
    the the the" is a perfect summary of any English sentence. Recall
    is what the spec line defines; precision and F1 are reported
    alongside because recall alone rewards padding the hypothesis.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, ROUGE (Lin 2004).

    Examples
    --------
    >>> out = kamath_rouge_n("the cat sat", "the cat sat on the mat", 1)
    >>> out["estimate"]
    0.5
    >>> out["precision"]
    1.0
    >>> bi = kamath_rouge_n("the cat sat", "the cat sat on the mat", 2)
    >>> abs(bi["estimate"] - 2 / 5) < 1e-12
    True
    >>> spam = kamath_rouge_n("the the the the", "the cat", 1)
    >>> abs(spam["estimate"] - 0.5) < 1e-12
    True
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    h, r = _toks(hypothesis), _toks(reference)
    if not r:
        raise ValueError("the reference is empty; recall would be 0/0.")
    if len(r) < n:
        raise ValueError(
            f"the reference has {len(r)} tokens, too few for "
            f"{n}-grams.")
    if not h:
        raise ValueError(
            "the hypothesis is empty; ROUGE is 0 by construction, "
            "which says nothing about the model.")
    hg, rg = _ngrams(h, n), _ngrams(r, n)
    total_r = sum(rg.values())
    total_h = sum(hg.values())
    match = sum(min(c, hg[g]) for g, c in rg.items())
    recall = match / total_r
    precision = match / total_h if total_h else 0.0
    f1 = (0.0 if recall + precision == 0
          else 2 * recall * precision / (recall + precision))
    return RichResult(payload={
        "estimate": recall, "recall": recall,
        "precision": precision, "f1": f1,
        "n_matched": match, "n_reference_ngrams": total_r,
        "n_hypothesis_ngrams": total_h,
        "order": n, "n": total_r,
        "method": f"ROUGE-{n} recall with clipped n-gram counts"})


def cheatsheet():
    return "kmroug: clipped n-gram matches / reference n-grams (+P, F1)"


# compact alias per ledger/NAMING.md
kamathrougen = kamath_rouge_n
