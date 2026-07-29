# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 8: BLEU end to end, from tokens to score."""

from collections import Counter

from ._richresult import RichResult
from .km114 import kamath_ch8_bleu_precision
from .km116 import kamath_ch8_brevity_penalty
from .km117 import kamath_ch8_bleu_final

__all__ = ["kamath_bleu_score"]


def _counts(tokens, n):
    return Counter(tuple(tokens[i:i + n])
                   for i in range(len(tokens) - n + 1))


def kamath_bleu_score(hypothesis, references, max_n=4):
    r"""BLEU = BP * exp(sum_n (1/N) log p_n) over a tokenized pair.

    What this adds to Eqs 8.2-8.5 is the COUNTING: clipped n-gram
    matches against the maximum count in any reference, and the
    effective reference length (the closest reference length to the
    candidate's, Papineni's rule). The three formulas themselves are
    delegated -- ``morie.fn.km114`` for p_n, ``km116`` for the brevity
    penalty and ``km117`` for the product.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, BLEU (Eqs 8.2-8.5);
    Papineni et al. (2002).

    Examples
    --------
    >>> import math
    >>> out = kamath_bleu_score(["the", "cat"],
    ...                         [["the", "cat", "sat"]], max_n=1)
    >>> abs(out["bleu"] - math.exp(1 - 3 / 2)) < 1e-12   # p_1 = 1
    True
    """
    hyp = list(hypothesis)
    refs = [list(r) for r in references]
    if isinstance(hypothesis, str) or any(isinstance(r, str)
                                          for r in references):
        raise ValueError("pass tokenized sequences, not raw strings.")
    if len(refs) == 0:
        raise ValueError("BLEU needs at least one reference.")
    if len(hyp) == 0:
        raise ValueError("the hypothesis is empty; BLEU's precisions "
                         "are 0/0 there.")
    N = int(max_n)
    if N < 1:
        raise ValueError(f"max_n must be at least 1; got {N}.")
    if len(hyp) < N:
        raise ValueError(
            f"the hypothesis has {len(hyp)} tokens, fewer than max_n = "
            f"{N}; the higher-order precisions are undefined.")
    pairs = []
    for n in range(1, N + 1):
        hc = _counts(hyp, n)
        best = Counter()
        for r in refs:
            rc = _counts(r, n)
            for g, c in rc.items():
                if c > best[g]:
                    best[g] = c
        clipped = sum(min(c, best[g]) for g, c in hc.items())
        pairs.append([clipped, sum(hc.values())])
    prec = kamath_ch8_bleu_precision(pairs)
    c = len(hyp)
    r_eff = min((abs(len(r) - c), len(r)) for r in refs)[1]
    bp = kamath_ch8_brevity_penalty(c, r_eff)
    final = kamath_ch8_bleu_final(bp["estimate"], prec["p_n"])
    return RichResult(payload={
        "estimate": final["estimate"], "bleu": final["estimate"],
        "p_n": prec["p_n"], "clipped_counts": pairs,
        "brevity_penalty": bp["estimate"], "candidate_length": c,
        "reference_length": r_eff, "max_n": N, "n": len(hyp),
        "method": "BLEU (Kamath Ch 8; km114/km116/km117 cores)"})


def cheatsheet():
    return "kmbleu: clipped n-gram counting feeding km114/km116/km117"
