# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WordPiece tokenizer: maximum likelihood subword segmentation."""

from collections import Counter

from ._richresult import RichResult

__all__ = ["geron_wordpiece_tokenizer"]

CONT = "##"


def _word_counts(corpus):
    if isinstance(corpus, str):
        corpus = [corpus]
    counts = Counter()
    for line in corpus:
        for w in str(line).split():
            if w:
                counts[w] += 1
    return counts


def _split(word):
    return [word[0]] + [CONT + c for c in word[1:]]


def geron_wordpiece_tokenizer(corpus, vocab_size=50):
    """
    WordPiece tokenizer: maximum likelihood subword segmentation.

    Formula: segment word into subwords maximizing likelihood under LM

    WordPiece and BPE differ in exactly one place, and it is implemented
    here rather than glossed over. BPE (see
    :func:`morie.fn.hmbpet.geron_bpe_tokenizer`) merges the *most
    frequent* pair. WordPiece merges the pair that most increases the
    unigram likelihood of the corpus, which after cancelling constant
    terms is the pair maximising

    ``score(A, B) = freq(AB) / (freq(A) * freq(B))``.

    The denominator is what makes the difference: a pair whose parts are
    common everywhere (``t`` + ``h``) scores badly even when the pair
    itself is frequent, while a pair whose parts occur almost only
    together scores highly. Continuation pieces carry the ``##`` prefix,
    and segmentation is greedy longest-match-first over the learned
    vocabulary; a word containing a character that never appeared in
    training is unsegmentable and reported as ``[UNK]``.

    Parameters
    ----------
    corpus : str or sequence of str
        Training text (whitespace tokenised into words).
    vocab_size : int, default 50
        Target vocabulary size; must be at least the alphabet size.

    Returns
    -------
    result : RichResult
        Keys: vocab, merges, scores, tokenize, alphabet, estimate,
        n, method.

    Examples
    --------
    "ug" occurs only inside "hug"/"hugs", so it is learned early even
    though other pairs are no rarer:

    >>> r = geron_wordpiece_tokenizer(["hug hug hugs pug pun"], vocab_size=14)
    >>> "##u" in r["alphabet"]
    True
    >>> toks = r["tokenize"]("hugs")
    >>> "".join(t.replace("##", "") for t in toks)
    'hugs'
    >>> bool(len(r["vocab"]) <= 14)
    True

    An unseen character cannot be segmented, and that is reported rather
    than silently dropped:

    >>> r["tokenize"]("zzz")
    ['[UNK]']

    References
    ----------
    Géron Ch 14
    """
    counts = _word_counts(corpus)
    if not counts:
        raise ValueError("geron_wordpiece_tokenizer: corpus contains no words")
    V = int(vocab_size)

    splits = {w: _split(w) for w in counts}
    alphabet = sorted({p for s in splits.values() for p in s})
    if V < len(alphabet):
        raise ValueError(
            f"geron_wordpiece_tokenizer: vocab_size {V} is smaller than the {len(alphabet)}-piece alphabet; "
            "every character must be representable"
        )
    vocab = list(alphabet)
    merges = []
    scores = []

    while len(vocab) < V:
        pair_freq = Counter()
        piece_freq = Counter()
        for w, c in counts.items():
            s = splits[w]
            for p in s:
                piece_freq[p] += c
            for i in range(len(s) - 1):
                pair_freq[(s[i], s[i + 1])] += c
        if not pair_freq:
            break
        best, best_score = None, -1.0
        for (a, b), f in sorted(pair_freq.items()):
            sc = f / (piece_freq[a] * piece_freq[b])
            if sc > best_score:
                best, best_score = (a, b), sc
        a, b = best
        new = a + b[len(CONT) :] if b.startswith(CONT) else a + b
        if new in vocab:
            break
        vocab.append(new)
        merges.append((a, b))
        scores.append(float(best_score))
        for w in splits:
            s = splits[w]
            out = []
            i = 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            splits[w] = out

    vocab_set = set(vocab)

    def tokenize(word, _vocab=vocab_set):
        """Greedy longest-match-first segmentation over the learned vocabulary."""
        w = str(word)
        if not w:
            raise ValueError("tokenize: the word is empty")
        out = []
        start = 0
        while start < len(w):
            end = len(w)
            piece = None
            while start < end:
                cand = w[start:end] if start == 0 else CONT + w[start:end]
                if cand in _vocab:
                    piece = cand
                    break
                end -= 1
            if piece is None:
                return ["[UNK]"]
            out.append(piece)
            start = end
        return out

    return RichResult(
        title="WordPiece tokenizer",
        summary_lines=[
            ("Words", int(sum(counts.values()))),
            ("Alphabet", len(alphabet)),
            ("Vocabulary", len(vocab)),
            ("Merges learned", len(merges)),
        ],
        interpretation=(
            "The likelihood score, not raw frequency, is what separates WordPiece from BPE: it prefers "
            "pairs whose parts are rare apart, which is why it recovers morpheme-like pieces."
        ),
        payload={
            "vocab": vocab,
            "merges": merges,
            "scores": scores,
            "tokenize": tokenize,
            "alphabet": alphabet,
            "word_counts": dict(counts),
            "estimate": float(len(vocab)),
            "n": int(sum(counts.values())),
            "method": "WordPiece: greedy likelihood-scored merges freq(AB)/(freq(A)freq(B)), longest-match segmentation",
        },
    )


def cheatsheet():
    return "hmwpt: WordPiece tokenizer: maximum likelihood subword segmentation"
