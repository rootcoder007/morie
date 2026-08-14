# morie.fn -- function file (rootcoder007/morie)
r"""SentencePiece: subword tokenisation that is losslessly reversible.

Most NMT pipelines still lean on language-specific pre- and
post-processors -- Moses and its hand-written rules -- which were built
for whitespace-delimited European languages. Chinese, Japanese and
Korean need a separate word segmenter first, and multilingual training
then means managing a different configuration per language while the
network itself is language-independent. SentencePiece removes that
asymmetry by training directly on raw sentences.

**Lossless tokenisation is the property everything rests on.** Whitespace
is not discarded; it is escaped as a visible character, U+2581 (a lower
one-eighth block, written ``_`` in the paper's examples). A sentence is
prefixed with it, spaces become it, and detokenisation is then a pure
string operation:

.. math:: \mathrm{Decode}(\mathrm{Encode}(x)) = x \quad
          \text{for every } x.

That is an identity, not a best effort, and it is what makes the
tokeniser safe to invert -- no language-specific detokenisation rules,
no lost double spaces. The anchor checks it on inputs chosen to break
naive schemes: leading spaces, runs of spaces, no spaces at all.

**Two segmentation algorithms, both offered.** BPE merges the most
frequent adjacent pair repeatedly until the vocabulary is full -- a
greedy, deterministic construction. The unigram language model instead
posits a probability for each piece and picks the segmentation
maximising :math:`\prod_i p(x_i)`, found by Viterbi over the lattice of
possible splits. They differ in kind: BPE gives one segmentation by
construction, while the unigram model scores *all* of them and can
therefore also sample alternatives.

**Vocabulary size is fixed in advance.** That is the design constraint
-- the neural model needs it before training -- and it is why the
trainer's job is to choose which pieces earn a slot rather than to
discover how many are needed.

References
----------
Kudo, T. & Richardson, J. (2018) "SentencePiece: A simple and language
independent subword tokenizer and detokenizer for Neural Text
Processing", *Proceedings of the 2018 Conference on Empirical Methods
in Natural Language Processing: System Demonstrations*, 66-71,
doi:10.18653/v1/D18-2012, arXiv:1808.06226. Sec. 1 (the dependence of NMT pipelines on
language-specific processors and the difficulty this creates for
non-segmented languages and multilingual models), Sec. 2 (Normalizer,
Trainer, Encoder, Decoder; the predetermined vocabulary size), and
Sec. 3.1 (lossless tokenisation with whitespace escaped as U+2581).
Both BPE and the unigram language model are named as the two
implemented subword algorithms.

Sennrich, R., Haddow, B. & Birch, A. (2016) "Neural Machine
Translation of Rare Words with Subword Units", *Proceedings of the
54th Annual Meeting of the Association for Computational Linguistics*,
1715-1725, arXiv:1508.07909. The BPE algorithm.

Kudo, T. (2018) "Subword Regularization: Improving Neural Network
Translation Models with Multiple Subword Candidates", *Proceedings of
the 56th Annual Meeting of the Association for Computational
Linguistics*, 66-75, arXiv:1804.10959. The unigram language model and
the sampling of alternative segmentations.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["escape_whitespace", "unescape_whitespace", "train_bpe",
           "encode_bpe", "viterbi_segment", "decode"]

_EPS = 1e-300
SPACE = "▁"


def escape_whitespace(text, add_prefix=True):
    r"""Replace spaces with U+2581, optionally prefixing one.

    Nothing is dropped, which is the whole point: the transform is
    invertible by string replacement alone.
    """
    s = str(text)
    out = s.replace(" ", SPACE)
    if add_prefix:
        out = SPACE + out
    return out


def unescape_whitespace(text, strip_prefix=True):
    r"""Invert :func:`escape_whitespace` exactly."""
    s = str(text)
    if strip_prefix and s.startswith(SPACE):
        s = s[1:]
    return s.replace(SPACE, " ")


def _units(escaped):
    r"""Split an escaped string so that every unit after the first
    begins with U+2581, and ``"".join(units)`` reproduces the input.

    Splitting on the marker and re-attaching one would silently
    collapse runs of spaces, which would break the lossless identity.
    """
    out, cur = [], ""
    for ch in escaped:
        if ch == SPACE:
            if cur:
                out.append(cur)
            cur = SPACE
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def decode(pieces, strip_prefix=True):
    r"""Join pieces and unescape -- a pure string operation.

    No language-specific rules, which is why it is the same code for
    English and Japanese.
    """
    return unescape_whitespace("".join(str(p) for p in pieces),
                               strip_prefix=strip_prefix)


def train_bpe(corpus, vocab_size, add_prefix=True):
    r"""Merge the most frequent adjacent pair until the budget is met.

    Greedy and deterministic: the merge list fully determines every
    future segmentation.
    """
    V = int(vocab_size)
    if V < 1:
        raise ValueError("sentpc: vocab_size must be at least 1")
    words = {}
    for line in corpus:
        for w in _units(escape_whitespace(line, add_prefix)):
            words[tuple(w)] = words.get(tuple(w), 0) + 1
    if not words:
        raise ValueError("sentpc: the corpus produced no tokens")
    alphabet = sorted({c for w in words for c in w})
    merges = []
    vocab = set(alphabet)
    while len(vocab) < V:
        pairs = {}
        for w, f in words.items():
            for i in range(len(w) - 1):
                pairs[(w[i], w[i + 1])] = \
                    pairs.get((w[i], w[i + 1]), 0) + f
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        merges.append(best)
        vocab.add(best[0] + best[1])
        nw = {}
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            nw[tuple(out)] = nw.get(tuple(out), 0) + f
        words = nw
    return {"merges": merges, "vocab": sorted(vocab),
            "vocab_size": len(vocab), "requested": V,
            "algorithm": "bpe",
            "note": "greedy and deterministic -- the merge list fixes "
                    "every later segmentation"}


def encode_bpe(text, model, add_prefix=True):
    r"""Apply the learned merges in order."""
    esc = escape_whitespace(text, add_prefix)
    out = []
    for w in _units(esc):
        toks = list(w)
        for a, b in model["merges"]:
            i = 0
            new = []
            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == a \
                        and toks[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(toks[i])
                    i += 1
            toks = new
        out.extend(toks)
    return out


def viterbi_segment(text, piece_logp, add_prefix=True):
    r"""The unigram LM segmentation maximising :math:`\prod_i p(x_i)`.

    Viterbi over the lattice of all splits. Unlike BPE this scores
    every segmentation, so the best one is a maximisation rather than
    the by-product of a construction.
    """
    s = escape_whitespace(text, add_prefix)
    n = len(s)
    if n == 0:
        return {"pieces": [], "logp": 0.0}
    maxlen = max((len(p) for p in piece_logp), default=1)
    best = [-math.inf] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for L in range(1, min(maxlen, i) + 1):
            piece = s[i - L:i]
            lp = piece_logp.get(piece)
            if lp is None:
                continue
            if best[i - L] + lp > best[i]:
                best[i] = best[i - L] + lp
                back[i] = (i - L, piece)
    if best[n] == -math.inf:
        raise ValueError("sentpc: no segmentation covers the input -- "
                         "the piece set must include every character")
    pieces, i = [], n
    while i > 0:
        j, p = back[i]
        pieces.append(p)
        i = j
    pieces.reverse()
    return {"pieces": pieces, "logp": best[n],
            "n_pieces": len(pieces),
            "algorithm": "unigram (Viterbi)"}


def cheatsheet():
    return ("sentpc: SentencePiece. Whitespace is ESCAPED as U+2581, "
            "not dropped, so Decode(Encode(x)) == x EXACTLY -- no "
            "language-specific detokeniser, which is what makes "
            "multilingual and non-segmented languages work. Two "
            "algorithms: BPE merges the most frequent pair greedily "
            "(one segmentation by construction), the unigram LM "
            "scores ALL segmentations and takes the Viterbi best (so "
            "it can also sample). Vocabulary size is fixed in "
            "advance, because the neural model needs it.")


# compact alias per ledger/NAMING.md
sentencepiece = viterbi_segment
