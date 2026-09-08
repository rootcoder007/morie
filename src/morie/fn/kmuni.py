# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unigram LM tokenizer: EM over subword piece probabilities."""

import math

from ._richresult import RichResult

__all__ = ["kamath_unigram_lm_tokenizer", "unigram_loglik",
           "viterbi_segment"]


def _pieces_by_end(text, vocab_set, maxlen):
    """For every position, the pieces of ``text`` ending there."""
    L = len(text)
    ends = [[] for _ in range(L + 1)]
    for j in range(1, L + 1):
        for n in range(1, min(maxlen, j) + 1):
            w = text[j - n:j]
            if w in vocab_set:
                ends[j].append(w)
    return ends


def _forward(text, probs, maxlen):
    """alpha[j] = total probability of all segmentations of text[:j]."""
    L = len(text)
    ends = _pieces_by_end(text, probs, maxlen)
    alpha = [0.0] * (L + 1)
    alpha[0] = 1.0
    for j in range(1, L + 1):
        tot = 0.0
        for w in ends[j]:
            tot += alpha[j - len(w)] * probs[w]
        alpha[j] = tot
    return alpha, ends


def _backward(text, probs, maxlen):
    L = len(text)
    beta = [0.0] * (L + 1)
    beta[L] = 1.0
    for j in range(L - 1, -1, -1):
        tot = 0.0
        for n in range(1, min(maxlen, L - j) + 1):
            w = text[j:j + n]
            if w in probs:
                tot += probs[w] * beta[j + n]
        beta[j] = tot
    return beta


def unigram_loglik(corpus, probs):
    """sum over sentences of log P(s) = log sum over segmentations of
    prod p(w). Raises if a sentence cannot be segmented at all."""
    if not probs:
        raise ValueError("the piece table is empty.")
    maxlen = max(len(w) for w in probs)
    total = 0.0
    for s in corpus:
        if not s:
            continue
        alpha, _ = _forward(s, probs, maxlen)
        if alpha[-1] <= 0.0:
            raise ValueError(
                f"{s!r} has no segmentation under this vocabulary; add "
                "the missing characters as single-character pieces.")
        total += math.log(alpha[-1])
    return total


def viterbi_segment(text, probs):
    """The single most probable segmentation (what the tokenizer
    actually emits at inference)."""
    maxlen = max(len(w) for w in probs)
    L = len(text)
    best = [-math.inf] * (L + 1)
    back = [None] * (L + 1)
    best[0] = 0.0
    for j in range(1, L + 1):
        for n in range(1, min(maxlen, j) + 1):
            w = text[j - n:j]
            p = probs.get(w)
            if p is None or p <= 0 or best[j - n] == -math.inf:
                continue
            cand = best[j - n] + math.log(p)
            if cand > best[j]:
                best[j] = cand
                back[j] = w
    if best[L] == -math.inf:
        raise ValueError(f"{text!r} cannot be segmented by this vocabulary.")
    out, j = [], L
    while j > 0:
        w = back[j]
        out.append(w)
        j -= len(w)
    return out[::-1], best[L]


def kamath_unigram_lm_tokenizer(corpus, vocab, max_iter=100, tol=1e-12):
    """E-step: expected counts of each piece over ALL segmentations;
    M-step: p(w) proportional to that expected count.

    The E-step is forward-backward, not Viterbi: a piece gets credit
    from every segmentation it appears in, weighted by that
    segmentation's posterior. Using the single best path instead is
    "hard EM" and gives a different fixed point -- the distinction the
    unigram model is built on.

    The log-likelihood after every iteration is returned; EM cannot
    decrease it, which makes the history a runnable check on the
    implementation rather than a decoration.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, unigram LM
    (Kudo 2018).

    Examples
    --------
    >>> out = kamath_unigram_lm_tokenizer(["ab"], ["a", "b"])
    >>> [round(out["probs"][w], 12) for w in ["a", "b"]]
    [0.5, 0.5]
    >>> import math
    >>> abs(out["log_likelihood"] - math.log(0.25)) < 1e-12
    True
    >>> hist = out["log_likelihood_history"]
    >>> all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))
    True
    >>> em = kamath_unigram_lm_tokenizer(["ab", "ab"], ["a", "b", "ab"])
    >>> em["probs"]["ab"] > 0.9
    True
    >>> em["segmentations"][0]
    ['ab']
    """
    corpus = [s for s in corpus if s]
    if not corpus:
        raise ValueError("the corpus is empty.")
    vocab = list(dict.fromkeys(vocab))
    if not vocab:
        raise ValueError("the vocabulary is empty.")
    if any(len(w) == 0 for w in vocab):
        raise ValueError("the empty string is not a subword piece.")
    maxlen = max(len(w) for w in vocab)
    probs = {w: 1.0 / len(vocab) for w in vocab}
    # Fail fast on an unsegmentable corpus rather than after 100
    # iterations of nan.
    unigram_loglik(corpus, probs)

    history = []
    for _ in range(int(max_iter)):
        counts = {w: 0.0 for w in vocab}
        ll = 0.0
        for s in corpus:
            alpha, ends = _forward(s, probs, maxlen)
            beta = _backward(s, probs, maxlen)
            Z = alpha[-1]
            ll += math.log(Z)
            for j in range(1, len(s) + 1):
                for w in ends[j]:
                    counts[w] += (alpha[j - len(w)] * probs[w]
                                  * beta[j]) / Z
        history.append(ll)
        total = sum(counts.values())
        if total <= 0:
            raise ValueError("the E-step produced no expected counts.")
        new = {w: c / total for w, c in counts.items()}
        delta = max(abs(new[w] - probs[w]) for w in vocab)
        probs = new
        if delta < tol:
            break
    final_ll = unigram_loglik(corpus, probs)
    history.append(final_ll)
    segs = [viterbi_segment(s, probs)[0] for s in corpus]
    return RichResult(payload={
        "probs": probs,
        "log_likelihood": final_ll,
        "log_likelihood_history": history,
        "n_iterations": len(history) - 1,
        "segmentations": segs,
        "vocab_size": len(vocab),
        "estimate": final_ll,
        "n": len(corpus),
        "method": "Unigram LM EM (forward-backward expected counts)"})


def cheatsheet():
    return "kmuni: EM over piece probs; soft counts, monotone log-likelihood"


# compact alias per ledger/NAMING.md
unigramloglik = unigram_loglik


# compact alias per ledger/NAMING.md
viterbisegment = viterbi_segment
