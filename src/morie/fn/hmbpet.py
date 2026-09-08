# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair."""

from collections import OrderedDict

from ._richresult import RichResult

__all__ = ["geron_bpe_tokenizer"]

_EOW = "</w>"


def _word_counts(corpus):
    counts = OrderedDict()
    if isinstance(corpus, dict):
        for word, c in corpus.items():
            if int(c) < 0:
                raise ValueError("geron_bpe_tokenizer: corpus counts must be non-negative")
            counts[str(word)] = counts.get(str(word), 0) + int(c)
        return counts
    if isinstance(corpus, str):
        words = corpus.split()
    else:
        words = []
        for item in corpus:
            words.extend(str(item).split())
    for wd in words:
        counts[wd] = counts.get(wd, 0) + 1
    return counts


def geron_bpe_tokenizer(corpus, vocab_size=100):
    """
    Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair.

    Formula: at each step merge argmax pair; rebuild vocab

    Ties on frequency are broken by first appearance in corpus order, so the
    merge sequence is fully deterministic.

    Parameters
    ----------
    corpus : str, sequence of str, or dict
        Whitespace-separated text, a list of strings, or a ``{word: count}``
        mapping.
    vocab_size : int
        Target vocabulary size, counting the base symbols. Merging stops when
        the vocabulary reaches it or when no pair occurs more than once.

    Returns
    -------
    result : RichResult
        Keys: merges, vocab, tokenize, word_tokens, n_merges, estimate, n, method.

    Examples
    --------
    The textbook corpus (low x5, lower x2, newest x6, widest x3) merges
    "e"+"s" first, then "es"+"t", then "est"+"</w>":

    >>> corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    >>> r = geron_bpe_tokenizer(corpus, vocab_size=14)
    >>> r["merges"]
    [('e', 's'), ('es', 't'), ('est', '</w>')]
    >>> r["tokenize"]("newest")
    ['n', 'e', 'w', 'est</w>']
    >>> r["tokenize"]("lowest")
    ['l', 'o', 'w', 'est</w>']

    Continuing to a larger vocabulary reproduces the rest of the textbook
    merge order and eventually swallows whole words:

    >>> geron_bpe_tokenizer(corpus, vocab_size=20)["merges"][3:6]
    [('l', 'o'), ('lo', 'w'), ('n', 'e')]
    >>> geron_bpe_tokenizer(corpus, vocab_size=20)["tokenize"]("newest")
    ['newest</w>']

    Stopping early gives fewer merges:

    >>> geron_bpe_tokenizer(corpus, vocab_size=13)["n_merges"]
    2

    References
    ----------
    Géron Ch 14
    """
    counts = _word_counts(corpus)
    if not counts:
        raise ValueError("geron_bpe_tokenizer: corpus is empty")
    target = int(vocab_size)
    if target < 1:
        raise ValueError("geron_bpe_tokenizer: vocab_size must be >= 1")

    words = {w: tuple(list(w) + [_EOW]) for w in counts}
    vocab = OrderedDict()
    for w, syms in words.items():
        for s in syms:
            vocab.setdefault(s, 0)
            vocab[s] += counts[w]
    if target < len(vocab):
        # Base symbols cannot be dropped; report the floor rather than lie
        # about hitting the requested size.
        merges = []
    else:
        merges = None

    if merges is None:
        merges = []
        while len(vocab) < target:
            pair_freq = OrderedDict()
            for w, syms in words.items():
                c = counts[w]
                for i in range(len(syms) - 1):
                    key = (syms[i], syms[i + 1])
                    pair_freq[key] = pair_freq.get(key, 0) + c
            if not pair_freq:
                break
            best = max(pair_freq.items(), key=lambda kv: kv[1])[1]
            if best < 2:
                break
            # First key (insertion order) attaining the max wins the tie.
            pair = next(k for k, v in pair_freq.items() if v == best)
            new_sym = pair[0] + pair[1]
            for w, syms in list(words.items()):
                out = []
                i = 0
                while i < len(syms):
                    if i + 1 < len(syms) and (syms[i], syms[i + 1]) == pair:
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                words[w] = tuple(out)
            merges.append(pair)
            vocab[new_sym] = best

    ordered_merges = list(merges)

    def tokenize(word, _merges=ordered_merges):
        syms = list(str(word)) + [_EOW]
        for a, b in _merges:
            out = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        return syms

    return RichResult(
        title="Byte-pair encoding",
        summary_lines=[("Merges", len(ordered_merges)), ("Vocabulary size", len(vocab))],
        payload={
            "merges": ordered_merges,
            "vocab": dict(vocab),
            "tokenize": tokenize,
            "word_tokens": {w: list(s) for w, s in words.items()},
            "n_merges": len(ordered_merges),
            "estimate": float(len(vocab)),
            "n": int(len(counts)),
            "method": "Byte-pair encoding with greedy most-frequent-pair merges",
        },
    )


def cheatsheet():
    return "hmbpet: Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair"
