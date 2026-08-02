# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Byte-pair encoding merge step: merge the most-frequent adjacent pair."""

from collections import Counter

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bpe_tokenizer_merge"]

_METHOD = "Byte-pair encoding merges"

_EOW = "</w>"


def geron_bpe_tokenizer_merge(corpus, n_merges):
    r"""Learn ``n_merges`` byte-pair-encoding merges from a corpus.

    Each word starts as its sequence of characters plus an end-of-word
    marker.  At every round

    .. math::
        \text{pair}^* = \arg\max_{\text{pair}} \operatorname{count}
        (\text{pair in corpus})

    and both symbols of that pair are glued into one.  Ties are broken by
    taking the lexicographically smallest pair, so the result is
    deterministic -- BPE implementations that iterate a hash map are not,
    and their vocabularies differ run to run.

    Merging stops early if no adjacent pair occurs more than once, since
    a merge with count 1 buys no compression.

    Parameters
    ----------
    corpus : sequence of str, or mapping str -> int
        Words, either as a list (repeats count) or as word/frequency
        pairs.
    n_merges : int
        Maximum number of merges to learn, non-negative.

    Returns
    -------
    RichResult
        Payload keys ``merges`` (ordered list of merged pairs),
        ``vocab`` (sorted final symbol inventory), ``splits`` (word to
        final token list), ``merge_counts``, ``n_tokens_before``,
        ``n_tokens_after``, ``compression``, ``estimate`` (number of
        merges actually performed), ``n``, ``method``.

    References
    ----------
    Géron Ch 14, BPE tokenizer section.

    Examples
    --------
    >>> r = geron_bpe_tokenizer_merge({"low": 5, "lowest": 2}, 2)
    >>> r["merges"]
    [('l', 'o'), ('lo', 'w')]
    >>> r["splits"]["low"]
    ['low', '</w>']
    >>> r["splits"]["lowest"]
    ['low', 'e', 's', 't', '</w>']
    >>> r["merge_counts"]
    [7, 7]
    """
    if isinstance(corpus, dict):
        freqs = {str(w): int(c) for w, c in corpus.items()}
    else:
        words = [str(w) for w in corpus]
        freqs = dict(Counter(words))
    if not freqs:
        raise ValueError("corpus is empty.")
    if any(c <= 0 for c in freqs.values()):
        raise ValueError("word frequencies must be positive.")
    if any(len(w) == 0 for w in freqs):
        raise ValueError("corpus contains an empty word.")
    n_merges = int(n_merges)
    if n_merges < 0:
        raise ValueError(f"n_merges must be non-negative, got {n_merges}.")

    splits = {w: list(w) + [_EOW] for w in freqs}
    n_before = sum(len(s) * freqs[w] for w, s in splits.items())

    merges = []
    counts = []
    for _ in range(n_merges):
        pair_counts = Counter()
        for w, syms in splits.items():
            f = freqs[w]
            for i in range(len(syms) - 1):
                pair_counts[(syms[i], syms[i + 1])] += f
        if not pair_counts:
            break
        best_count = max(pair_counts.values())
        if best_count < 2:
            break
        best = min(p for p, c in pair_counts.items() if c == best_count)
        merged = best[0] + best[1]
        for w, syms in splits.items():
            out = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            splits[w] = out
        merges.append(best)
        counts.append(int(best_count))

    n_after = sum(len(s) * freqs[w] for w, s in splits.items())
    vocab = sorted({s for syms in splits.values() for s in syms})

    return RichResult(
        title="BPE merges",
        summary_lines=[("Merges learned", len(merges)), ("Vocabulary size", len(vocab))],
        payload={
            "merges": merges,
            "vocab": vocab,
            "splits": splits,
            "merge_counts": counts,
            "n_tokens_before": int(n_before),
            "n_tokens_after": int(n_after),
            "compression": float(n_before / n_after) if n_after else float("nan"),
            "estimate": float(len(merges)),
            "n": int(len(freqs)),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbpe: BPE -- repeatedly merge the most frequent adjacent symbol pair (ties: lexicographic)"
