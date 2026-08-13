# morie.fn -- function file (rootcoder007/morie)
r"""FastText -- word vectors enriched with subword information.

Bojanowski, Grave, Joulin & Mikolov (2017). A word is represented as a
bag of character n-grams, each with its own vector, and the scoring
function of Sec. 3.2 sums over them:

.. math:: s(w, c) = \sum_{g \in \mathcal{G}_w} z_g^{\top} v_c,

so a word's representation is the sum of its n-gram vectors. The stub
returned ``mean(corpus)``.

**The boundary symbols are the point.** The paper adds ``<`` and ``>``
around every word before extracting n-grams, so that a prefix or suffix
is distinguishable from the same letters in the middle of a word, and
it keeps the whole word as a further unit. Their own worked example,
which the anchor checks character for character: *where* with n = 3
gives

    <wh, whe, her, ere, re>

plus the special sequence ``<where>``. Note that ``her`` here is the
trigram from the middle of *where* and is **different from** the
n-gram ``<her>`` from the word *her* -- that distinction is exactly
what the boundary symbols buy, and dropping them silently merges the
two.

The default range is 3 to 6 inclusive, the paper's choice.

This module builds the subword decomposition and the vector sum, and
trains with skipgram negative sampling as in Sec. 3.1. It does not
reimplement the hashing trick of Sec. 3.3 (n-grams bucketed by a hash
into a fixed table) because that is a memory optimisation which changes
results through collisions; ``hash_buckets`` enables it explicitly for
callers who want the reference behaviour.

References
----------
Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017) "Enriching
word vectors with subword information", *Transactions of the
Association for Computational Linguistics* 5, 135-146,
doi:10.1162/tacl_a_00051; arXiv:1607.04606. Sec. 3.2 for the subword
model and the *where* example, Sec. 3.1 for the objective.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013)
"Distributed representations of words and phrases and their
compositionality", *Advances in Neural Information Processing Systems*
26, 3111-3119 -- the skipgram with negative sampling this extends.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["fasttext", "subwords", "word_vector"]


def subwords(word, n_min=3, n_max=6, boundary=True, whole_word=True):
    """The bag of character n-grams for one word, Sec. 3.2.

    Returns them in order of appearance, deduplicated. With the paper's
    defaults, ``subwords("where", 3, 3)`` is
    ``["<wh", "whe", "her", "ere", "re>", "<where>"]``.
    """
    lo, hi = int(n_min), int(n_max)
    if lo < 1:
        raise ValueError("subwords: n_min must be at least 1, got %r"
                         % (n_min,))
    if hi < lo:
        raise ValueError("subwords: n_max (%r) is below n_min (%r)"
                         % (n_max, n_min))
    w = str(word)
    padded = "<" + w + ">" if boundary else w
    grams, seen = [], set()
    for n in range(lo, hi + 1):
        for i in range(0, len(padded) - n + 1):
            g = padded[i:i + n]
            if g not in seen:
                seen.add(g)
                grams.append(g)
    if whole_word:
        special = "<" + w + ">" if boundary else w
        if special not in seen:
            grams.append(special)
    return grams


def word_vector(word, Z, gram_index, n_min=3, n_max=6, boundary=True,
                whole_word=True, hash_buckets=None):
    """Sum of the word's n-gram vectors -- the representation itself.

    A word made entirely of unseen n-grams gets a zero vector rather
    than an error: that is the out-of-vocabulary case fastText exists
    to handle, and returning zeros says "no information" honestly.
    """
    dim = len(Z[0]) if Z else 0
    v = [0.0] * dim
    hit = 0
    for g in subwords(word, n_min, n_max, boundary, whole_word):
        idx = _gram_slot(g, gram_index, hash_buckets)
        if idx is None:
            continue
        hit += 1
        for t in range(dim):
            v[t] += Z[idx][t]
    return v, hit


def _gram_slot(g, gram_index, hash_buckets):
    if hash_buckets:
        return _fnv1a(g) % int(hash_buckets)
    return gram_index.get(g)


def _fnv1a(s):
    """FNV-1a, the hash the reference implementation uses."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def fasttext(corpus, dim=50, n_min=3, n_max=6, window=5, epochs=5,
             lr=0.05, negative=5, min_count=1, boundary=True,
             whole_word=True, hash_buckets=None, seed=0):
    r"""Train subword word vectors by skipgram with negative sampling.

    Parameters
    ----------
    corpus : sequence
        Documents, each a token list or a whitespace-separated string.
    n_min, n_max : int
        Character n-gram range, inclusive. The paper uses 3 to 6.
    hash_buckets : int, optional
        Bucket the n-grams into a fixed table by FNV-1a, as Sec. 3.3
        does. Off by default because collisions change results.

    Returns
    -------
    RichResult
        ``estimate`` is the word-vector matrix in ``vocab`` order,
        built as the sum of each word's n-gram vectors.

    Examples
    --------
    An unseen word still gets a vector, from its n-grams::

        r = fasttext([["running", "runner", "ran"]], dim=8)
        r["oov"]("runs")
    """
    docs = _as_docs(corpus)
    d = int(dim)
    if d < 1:
        raise ValueError("fasttext: dim must be at least 1, got %r"
                         % (dim,))
    counts = {}
    for doc in docs:
        for t in doc:
            counts[t] = counts.get(t, 0) + 1
    vocab = sorted(t for t, c in counts.items() if c >= int(min_count))
    if len(vocab) < 2:
        raise ValueError(
            "fasttext: %d word(s) above min_count=%r; skipgram needs a "
            "context to predict" % (len(vocab), min_count))
    windex = {t: i for i, t in enumerate(vocab)}

    grams = []
    gram_index = {}
    for wd in vocab:
        for g in subwords(wd, n_min, n_max, boundary, whole_word):
            if g not in gram_index:
                gram_index[g] = len(grams)
                grams.append(g)
    n_slots = int(hash_buckets) if hash_buckets else len(grams)

    rng = np.random.default_rng(int(seed))
    sc = 0.5 / d
    Z = [[(float(rng.uniform()) - 0.5) * sc for _ in range(d)]
         for _ in range(n_slots)]          # n-gram (input) vectors
    Vc = [[0.0] * d for _ in range(len(vocab))]   # context vectors

    # negative sampling distribution: unigram^(3/4), Mikolov et al.
    freqs = [counts[t] ** 0.75 for t in vocab]
    tot = sum(freqs)
    cum, run = [], 0.0
    for f in freqs:
        run += f / tot
        cum.append(run)

    def draw_negative():
        u = float(rng.uniform())
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u > cum[mid]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    eta = float(lr)
    losses = []
    for _ in range(int(epochs)):
        total, n_upd = 0.0, 0
        for doc in docs:
            ids = [t for t in doc if t in windex]
            for pos, wd in enumerate(ids):
                slots = [s for s in
                         (_gram_slot(g, gram_index, hash_buckets)
                          for g in subwords(wd, n_min, n_max, boundary,
                                            whole_word))
                         if s is not None]
                if not slots:
                    continue
                u = [sum(Z[s][t] for s in slots) for t in range(d)]
                lo = max(0, pos - int(window))
                hi = min(len(ids), pos + int(window) + 1)
                for other in range(lo, hi):
                    if other == pos:
                        continue
                    targets = [(windex[ids[other]], 1.0)]
                    for _ in range(int(negative)):
                        targets.append((draw_negative(), 0.0))
                    grad_u = [0.0] * d
                    for ci, label in targets:
                        dot = sum(u[t] * Vc[ci][t] for t in range(d))
                        p = 1.0 / (1.0 + math.exp(-max(-30.0,
                                                       min(30.0, dot))))
                        g = (p - label)
                        total += -(math.log(p + 1e-12) if label > 0.5
                                   else math.log(1.0 - p + 1e-12))
                        n_upd += 1
                        for t in range(d):
                            grad_u[t] += g * Vc[ci][t]
                            Vc[ci][t] -= eta * g * u[t]
                    for s in slots:
                        for t in range(d):
                            Z[s][t] -= eta * grad_u[t]
        losses.append(total / n_upd if n_upd else float("nan"))

    vecs = []
    for wd in vocab:
        v, _ = word_vector(wd, Z, gram_index, n_min, n_max, boundary,
                           whole_word, hash_buckets)
        vecs.append(v)

    def oov(word):
        """The vector for any word, seen or not -- the point of fastText."""
        return word_vector(word, Z, gram_index, n_min, n_max, boundary,
                           whole_word, hash_buckets)[0]

    return RichResult(payload={
        "estimate": vecs,
        "vectors": vecs,
        "vocab": vocab,
        "index": windex,
        "ngrams": grams,
        "ngram_index": gram_index,
        "Z": Z, "context": Vc,
        "loss_history": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "oov": oov,
        "n_vocab": len(vocab), "n_ngrams": len(grams), "dim": d,
        "n_min": int(n_min), "n_max": int(n_max),
        "hash_buckets": hash_buckets,
        "method": "fastText subword skipgram with negative sampling, "
                  "Bojanowski, Grave, Joulin & Mikolov (2017) Sec. 3.2",
    })


def _as_docs(corpus):
    if corpus is None:
        raise ValueError("fasttext: corpus must not be None")
    docs = []
    for item in corpus:
        if isinstance(item, str):
            docs.append(item.split())
        else:
            docs.append([str(t) for t in item])
    if not docs:
        raise ValueError("fasttext: the corpus is empty")
    return docs


def cheatsheet():
    return ("fastxt: word = bag of character n-grams with < > "
            "boundaries plus the whole word; s(w,c) = sum_g z_g . v_c "
            "(Bojanowski et al. 2017 Sec.3.2). where/n=3 -> <wh whe her "
            "ere re> + <where>. n in 3..6. Gives OOV words a vector.")
