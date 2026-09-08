# morie.fn -- slice s03 (rootcoder007/morie)
"""Byte-pair-encoding subword tokenizer.

Source consulted (FETCHED): Sennrich, R., Haddow, B. and Birch, A.
(2016).  Neural machine translation of rare words with subword units.
*ACL* 54, 1715-1725 (arXiv:1508.07909), algorithm 1.  The training loop
is: represent every word as a sequence of characters plus an
end-of-word marker, count all adjacent symbol pairs across the corpus,
merge the most frequent pair into a new symbol, and repeat for the
requested number of merges.  The paper's own marker is ``</w>``, which
is used here.

DETERMINISM.  Ties in the pair counts are broken by the pair's first
appearance in a fixed scan order, not arbitrarily -- the merge list is
therefore reproducible, which is exactly what a tokenizer needs and what
lets the two arms agree.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["bpe_tokenizer"]

_EOW = "</w>"


def bpe_tokenizer(corpus, vocab_size=10, word_counts=None):
    """Learn a BPE merge table from a corpus.

    Parameters
    ----------
    corpus : list of str
        Words (repeats allowed), or the unique word list when
        ``word_counts`` is given.
    vocab_size : int
        Number of merge operations to learn.
    word_counts : list of float, optional
        Frequency of each word in ``corpus``.

    Returns
    -------
    RichResult with payload:
        estimate  : number of merges learned
        merges    : the merge pairs in learning order
        counts    : the count each merge had when applied
        vocab     : the final symbol inventory, sorted
        tokens    : the final segmentation of each word
    """
    words = [str(w) for w in corpus]
    if word_counts is not None:
        freq = [float(c) for c in word_counts]
    else:
        uniq = []
        cnt = []
        for w in words:
            if w in uniq:
                cnt[uniq.index(w)] += 1.0
            else:
                uniq.append(w)
                cnt.append(1.0)
        words = uniq
        freq = cnt
    seqs = [list(w) + [_EOW] for w in words]
    merges = []
    counts = []
    for _ in range(int(vocab_size)):
        pairs = []
        pc = []
        for wi in range(len(seqs)):
            s = seqs[wi]
            for j in range(len(s) - 1):
                p = (s[j], s[j + 1])
                if p in pairs:
                    pc[pairs.index(p)] += freq[wi]
                else:
                    pairs.append(p)
                    pc.append(freq[wi])
        if not pairs:
            break
        best = 0
        for i in range(1, len(pairs)):
            if pc[i] > pc[best]:
                best = i
        if pc[best] <= 1.0:
            break
        a, b = pairs[best]
        merges.append(a + "|" + b)
        counts.append(pc[best])
        for wi in range(len(seqs)):
            s = seqs[wi]
            out = []
            j = 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            seqs[wi] = out
        # symbols cannot be merged into the same string twice
    vocab = []
    for s in seqs:
        for sym in s:
            if sym not in vocab:
                vocab.append(sym)
    vocab = sorted(vocab)
    return RichResult(
        title="Byte-pair encoding",
        summary_lines=[("merges", len(merges)), ("vocab", len(vocab))],
        payload={
            "estimate": float(len(merges)),
            "merges": merges,
            "counts": counts,
            "vocab": vocab,
            "n_vocab": len(vocab),
            "tokens": seqs,
            "method": "BPE subword learner (Sennrich et al. 2016, algorithm 1)",
        },
    )


def cheatsheet():
    return "bpetk: Byte-pair encoding tokenizer"


bpetokenizer = bpe_tokenizer
