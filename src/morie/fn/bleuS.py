# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""BLEU: n-gram precision score for machine translation.

Source: Papineni, K., Roukos, S., Ward, T. and Zhu, W.-J. (2002),
"BLEU: a method for automatic evaluation of machine translation",
Proceedings of the 40th Annual Meeting of the ACL, 311-318, read from
the ACL Anthology PDF (P02-1040).  Sections 2.1-2.3 give, verbatim:

    p_n = sum_C sum_{ngram in C} Count_clip(ngram)
          / sum_C2 sum_{ngram2 in C2} Count(ngram2)

    BP = 1                if c > r
       = exp(1 - r/c)     if c <= r

    BLEU = BP * exp( sum_{n=1..N} w_n log p_n ),   N = 4, w_n = 1/N.

Two details carry the whole method and are easy to lose.

Clipping.  Count_clip is the candidate count truncated at the largest
count the n-gram attains in any single reference.  Without it the
paper's own Example 2 -- candidate "the the the the the the the" against
"The cat is on the mat." -- scores a unigram precision of 7/7 instead of
the 2/7 the paper prints.  That printed 2/7 is this module's anchor.

The brevity penalty is one-sided.  Candidates longer than the reference
are already punished by precision, so BP only bites when c <= r; r is
the "best match length", the reference length closest to c, ties going
to the shorter.  Making BP two-sided would double-penalise.

Case folding is the only normalisation the paper performs, and the only
one done here.  A zero at any order sends the geometric mean to zero,
and BLEU is honestly reported as 0 rather than smoothed; the per-order
precisions are returned so the cause is visible.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["bleu"]


def tokenize(s):
    """Whitespace tokens, case folded -- the paper's only normalisation."""
    if isinstance(s, str):
        return s.lower().split()
    return [str(v).lower() for v in s]


def ngram_counts(toks, n):
    """Multiset of n-grams as a dict keyed by the joined tokens."""
    d = {}
    for i in range(len(toks) - n + 1):
        g = " ".join(toks[i:i + n])
        d[g] = d.get(g, 0) + 1
    return d


def modified_precision(cand, refs, n):
    """Clipped numerator and total denominator for order n."""
    cc = ngram_counts(cand, n)
    tot = 0
    for v in cc.values():
        tot += v
    if tot == 0:
        return 0, 0
    mx = {}
    for r in refs:
        rc = ngram_counts(r, n)
        for g, v in rc.items():
            if v > mx.get(g, 0):
                mx[g] = v
    clip = 0
    for g, v in cc.items():
        m = mx.get(g, 0)
        clip += v if v < m else m
    return clip, tot


def best_match_length(c, ref_lens):
    """The reference length closest to c; ties go to the shorter one."""
    best = None
    for r in sorted(ref_lens):
        if best is None or abs(r - c) < abs(best - c):
            best = r
    return best


def bleu(candidate, references, max_n=4):
    """BLEU of one candidate against one or more references.

    Parameters
    ----------
    candidate : str or sequence of tokens
        The candidate translation.
    references : sequence
        One reference or a list of them.
    max_n : int
        Highest n-gram order, N in the paper; 4 by default.

    Returns
    -------
    bleu : the score in [0, 1]
    p_n : the modified precision at each order
    bp : the brevity penalty
    c, r : candidate length and best match length
    """
    N = int(max_n)
    if N < 1:
        raise ValueError("bleu: max_n must be at least one")
    cand = tokenize(candidate)
    if not cand:
        raise ValueError("bleu: the candidate is empty")
    if isinstance(references, str):
        refs = [tokenize(references)]
    else:
        rl = list(references)
        if not rl:
            raise ValueError("bleu: no references given")
        refs = [tokenize(r) for r in rl]
    for r in refs:
        if not r:
            raise ValueError("bleu: a reference is empty")
    pn = []
    num = []
    den = []
    for n in range(1, N + 1):
        a, b = modified_precision(cand, refs, n)
        num.append(a)
        den.append(b)
        pn.append((a + 0.0) / b if b > 0 else 0.0)
    c = len(cand)
    r = best_match_length(c, [len(x) for x in refs])
    bp = 1.0 if c > r else math.exp(1.0 - (r + 0.0) / c)
    w = 1.0 / N
    if min(pn) <= 0.0:
        sc = 0.0
        logsum = float("-inf")
    else:
        logsum = 0.0
        for p in pn:
            logsum += w * math.log(p)
        sc = bp * math.exp(logsum)
    return RichResult(
        title="BLEU",
        summary_lines=[("bleu", sc), ("bp", bp)],
        payload={
            "bleu": sc,
            "estimate": sc,
            "p_n": pn,
            "clipped": num,
            "total": den,
            "bp": bp,
            "c": c,
            "r": r,
            "log_geo_mean": logsum,
            "max_n": N,
            "n_ref": len(refs),
            "method": "Papineni et al. (2002) BLEU, clipped n-gram precision, one-sided brevity penalty",
        },
    )


def cheatsheet():
    return "bleuS: BLEU n-gram precision score"
