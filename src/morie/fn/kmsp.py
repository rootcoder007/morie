# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SentencePiece unigram tokenizer: seed a vocabulary, EM-fit it, then
prune to size by likelihood loss."""

from ._richresult import RichResult
from .kmuni import (kamath_unigram_lm_tokenizer, unigram_loglik,
                    viterbi_segment)

__all__ = ["kamath_sentencepiece_tokenizer"]


def _seed_vocab(corpus, max_piece_len, seed_size):
    """All single characters (mandatory -- without them a sentence can
    become unsegmentable) plus the most frequent longer substrings,
    scored by frequency * length as SentencePiece does."""
    chars, freq = set(), {}
    for s in corpus:
        chars.update(s)
        L = len(s)
        for i in range(L):
            for n in range(2, min(max_piece_len, L - i) + 1):
                w = s[i:i + n]
                freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: (-freq[w] * len(w), w))
    room = max(0, seed_size - len(chars))
    return sorted(chars) + ranked[:room]


def kamath_sentencepiece_tokenizer(corpus, vocab_size, max_piece_len=8,
                                   seed_multiplier=4, shrink=0.75,
                                   max_iter=50):
    """argmax over V and p of sum_s log P(s | V, p), with
    P(s) = sum over segmentations of prod_t p(w_t).

    Three stages, as in the paper: seed a large vocabulary, fit p by
    EM (DELEGATED to ``morie.fn.kmuni`` -- the E-step is the same
    forward-backward), then drop the pieces whose removal costs the
    corpus log-likelihood least, and refit. The pruning loss is
    computed EXACTLY (refit-free: the likelihood is recomputed with
    the piece deleted), not approximated by its expected count, and
    single characters are never dropped, because losing one makes some
    sentence unsegmentable.

    The pruning is greedy, exactly as published: each piece's loss is
    measured against the CURRENT probabilities, with no refit per
    candidate. That means the final vocabulary is a local optimum and
    can be beaten -- on the example below the returned vocabulary
    scores -6.65 while {a, b, c, "ab"} scores -2.70. The
    ``log_likelihood`` is returned so a caller can see that, rather
    than being told the answer is optimal.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, SentencePiece
    (Kudo and Richardson 2018).

    Examples
    --------
    >>> corpus = ["abab", "abab", "abc"]
    >>> out = kamath_sentencepiece_tokenizer(corpus, 4)
    >>> out["vocab_size"]
    4
    >>> sorted(set("abc") - set(out["vocab"]))
    []
    >>> out["log_likelihood"] <= 0
    True
    >>> "abab" in out["vocab"]
    True
    >>> out["segmentations"][0]
    ['abab']
    >>> better = {"a": 1e-9, "b": 1e-9, "c": 1 / 6, "ab": 5 / 6 - 2e-9}
    >>> unigram_loglik(corpus, better) > out["log_likelihood"]
    True
    """
    corpus = [s for s in corpus if s]
    if not corpus:
        raise ValueError("the corpus is empty.")
    vocab_size = int(vocab_size)
    chars = sorted({c for s in corpus for c in s})
    if vocab_size < len(chars):
        raise ValueError(
            f"vocab_size {vocab_size} is below the {len(chars)} distinct "
            "characters in the corpus; every character must stay in the "
            "vocabulary or some sentence becomes unsegmentable.")
    if max_piece_len < 2:
        raise ValueError("max_piece_len must be at least 2.")
    if not 0.0 < shrink < 1.0:
        raise ValueError(f"shrink must lie in (0, 1); got {shrink}.")

    vocab = _seed_vocab(corpus, max_piece_len,
                        max(vocab_size, int(seed_multiplier * vocab_size)))
    fit = kamath_unigram_lm_tokenizer(corpus, vocab, max_iter=max_iter)
    probs = dict(fit["probs"])
    rounds = 0
    while len(probs) > vocab_size:
        rounds += 1
        base = unigram_loglik(corpus, probs)
        losses = {}
        for w in probs:
            if len(w) == 1:
                continue
            trimmed = {k: v for k, v in probs.items() if k != w}
            s = sum(trimmed.values())
            trimmed = {k: v / s for k, v in trimmed.items()}
            try:
                losses[w] = base - unigram_loglik(corpus, trimmed)
            except ValueError:
                losses[w] = float("inf")   # removal breaks segmentability
        if not losses:
            raise ValueError(
                "only single-character pieces remain but the vocabulary "
                "is still too large; raise vocab_size.")
        keep_n = max(vocab_size, int(len(probs) * shrink))
        n_drop = min(len(losses), len(probs) - keep_n)
        if n_drop <= 0:
            n_drop = 1
        drop = sorted(losses, key=lambda w: (losses[w], w))[:n_drop]
        vocab = [w for w in probs if w not in set(drop)]
        fit = kamath_unigram_lm_tokenizer(corpus, vocab, max_iter=max_iter)
        probs = dict(fit["probs"])
    segs = [viterbi_segment(s, probs)[0] for s in corpus]
    return RichResult(payload={
        "vocab": sorted(probs, key=lambda w: (-probs[w], w)),
        "probs": probs,
        "vocab_size": len(probs),
        "log_likelihood": float(unigram_loglik(corpus, probs)),
        "segmentations": segs,
        "n_prune_rounds": rounds,
        "estimate": len(probs),
        "n": len(corpus),
        "method": "SentencePiece unigram: seed, EM (kmuni), exact-loss prune"})


def cheatsheet():
    return "kmsp: seed -> EM via kmuni -> prune by exact likelihood loss"
