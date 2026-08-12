r"""word2vec: the CBOW and continuous skip-gram architectures.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013) "Efficient
Estimation of Word Representations in Vector Space",
arXiv:1301.3781.

Both architectures are **log-linear**: the expensive non-linear hidden
layer of a neural language model is removed, leaving a projection into
:math:`D` dimensions and a softmax output. That is the whole reason
they scale, and it is why the vectors can be trained on a billion
words.

**CBOW** (section 3.1) predicts the current word from its context. The
projection layer is shared across all positions, so *all context words
project to the same place -- their vectors are averaged*, and word order
does not influence the projection (hence "bag of words"; "continuous"
because the representation is a dense vector rather than a count). The
paper's best configuration uses four history and four future words.
Training complexity (eq. 4):

.. math:: Q = N \times D + D \times \log_2 V.

**Skip-gram** (section 3.2) inverts it: each current word is the input
to a log-linear classifier predicting words within a range before and
after it. Complexity (eq. 5):

.. math:: Q = C \times (D + D \times \log_2 V).

with :math:`C` the maximum distance. The paper is explicit about how
the range is used, and it is not a fixed window: "*since the more
distant words are usually less related to the current word than those
close to it, we give less weight to the distant words by sampling less
from those words*" -- concretely, "*for each training word we will
select randomly a number* :math:`R` *in range* :math:`\langle 1; C
\rangle`, *and then use* :math:`R` *words from history and* :math:`R`
*from the future*". A word at distance :math:`d` is therefore used with
probability :math:`(C - d + 1)/C`, which is a triangular weighting
achieved by sampling rather than by an explicit weight. Implementing a
fixed window instead is the commonest way to get skip-gram subtly
wrong, so ``dynamic_window`` defaults to the paper's behaviour and the
sampling probability is checked in the anchors.

The output layer here is the full softmax. The paper's complexity
terms are written with :math:`\log_2 V`, reflecting the hierarchical
softmax it uses at scale; :func:`training_complexity` reports the
paper's :math:`Q` for either. Negative sampling is *not* from this
paper -- it appears in the follow-up (Mikolov et al., NIPS 2013) --
so it is not implemented here rather than being attributed to a paper
that does not contain it.

What the vectors are for is section 4's observation that they encode
regularities as *offsets*: ``vector("King") - vector("Man") +
vector("Woman")`` lands near ``vector("Queen")``.
:func:`analogy` performs that query, excluding the three question
words from the answer as the paper's evaluation does.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wrd2v", "word2vec", "analogy", "training_complexity"]

_ARCH = ("skip-gram", "cbow")


def _softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def training_complexity(architecture, D, V, N=None, C=None,
                        hierarchical=True):
    r"""The paper's :math:`Q` for one training example (eqs. 4-5).

    ``N`` is the number of context words for CBOW; ``C`` the maximum
    distance for skip-gram. With ``hierarchical=False`` the output
    term is :math:`V` rather than :math:`\log_2 V`, which is the cost
    of the plain softmax this module actually evaluates.
    """
    if architecture not in _ARCH:
        raise ValueError("wrd2v: architecture must be one of %r, got %r"
                         % (_ARCH, architecture))
    out = math.log(V, 2) if hierarchical else float(V)
    if architecture == "cbow":
        if N is None:
            raise ValueError("wrd2v: CBOW complexity needs N")
        return float(N) * D + D * out
    if C is None:
        raise ValueError("wrd2v: skip-gram complexity needs C")
    return float(C) * (D + D * out)


def wrd2v(corpus, size=16, window=5, architecture="skip-gram", lr=0.05,
          epochs=20, min_count=1, dynamic_window=True, seed=0):
    r"""Train word vectors with CBOW or continuous skip-gram.

    Parameters
    ----------
    corpus : sequence
        Sentences, each a sequence of tokens. Order matters within a
        sentence; sentences are independent.
    size : int
        Vector dimension :math:`D`.
    window : int
        :math:`C`, the maximum distance. The paper uses 5 for
        skip-gram and 4 history + 4 future for CBOW.
    architecture : {"skip-gram", "cbow"}
        Section 3.2 or section 3.1.
    lr : float
        Learning rate for stochastic gradient descent, which is what
        the paper trains with.
    epochs : int
        Passes over the corpus (:math:`E` of eq. 1).
    min_count : int
        Discard words rarer than this.
    dynamic_window : bool
        Draw :math:`R \sim \mathrm{Unif}\{1, \dots, C\}` per centre
        word, as section 3.2 specifies. Setting it False gives a fixed
        window, which is *not* what the paper describes.
    seed : int
        Seed for initialisation and window sampling.

    Returns
    -------
    RichResult
        ``estimate`` / ``vectors`` maps word to its input (projection)
        vector -- the word embedding. ``output_vectors`` are the
        softmax output-layer rows; ``vocab`` the retained vocabulary
        with counts; ``loss_curve`` the mean negative log-likelihood
        per epoch; ``similarity`` and ``most_similar`` are callables
        for cosine queries.

    References
    ----------
    Mikolov, Chen, Corrado & Dean (2013) arXiv:1301.3781, sections
    3.1-3.2 and eqs. 4-5.
    """
    if architecture not in _ARCH:
        raise ValueError("wrd2v: architecture must be one of %r, got %r"
                         % (_ARCH, architecture))
    size = int(size)
    window = int(window)
    if size < 1:
        raise ValueError("wrd2v: size must be >= 1")
    if window < 1:
        raise ValueError("wrd2v: window must be >= 1")

    sents = [list(s) for s in corpus]
    if not sents or not any(sents):
        raise ValueError("wrd2v: corpus must contain at least one "
                         "non-empty sentence")
    counts = {}
    for s in sents:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    vocab = sorted([w for w, c in counts.items() if c >= int(min_count)],
                   key=repr)
    if not vocab:
        raise ValueError("wrd2v: min_count = %r discarded every word"
                         % (min_count,))
    idx = dict((w, i) for i, w in enumerate(vocab))
    V = len(vocab)
    sents = [[w for w in s if w in idx] for s in sents]

    rng = np.random.default_rng(seed)
    scale = 0.5 / size
    W = [[(rng.random() * 2.0 - 1.0) * scale for _ in range(size)]
         for _ in range(V)]        # projection (input) vectors
    O = [[0.0] * size for _ in range(V)]     # output layer

    curve = []
    for _ep in range(max(1, int(epochs))):
        total = 0.0
        n_ex = 0
        for s in sents:
            L = len(s)
            for t in range(L):
                R = (1 + int(rng.random() * window) if dynamic_window
                     else window)
                lo = max(0, t - R)
                hi = min(L, t + R + 1)
                ctx = [idx[s[k]] for k in range(lo, hi) if k != t]
                if not ctx:
                    continue
                c = idx[s[t]]
                if architecture == "cbow":
                    total += _cbow_step(W, O, ctx, c, size, V, lr)
                    n_ex += 1
                else:
                    for j in ctx:
                        total += _sg_step(W, O, c, j, size, V, lr)
                        n_ex += 1
        curve.append(total / n_ex if n_ex else 0.0)

    vectors = dict((vocab[i], list(W[i])) for i in range(V))
    outv = dict((vocab[i], list(O[i])) for i in range(V))

    def similarity(a, b):
        return _cos(vectors[a], vectors[b])

    def most_similar(word, topn=5):
        if word not in vectors:
            raise KeyError("wrd2v: %r is not in the vocabulary" % (word,))
        sims = [(w, _cos(vectors[word], v)) for w, v in vectors.items()
                if w != word]
        sims.sort(key=lambda p: -p[1])
        return sims[:topn]

    return RichResult(payload={
        "estimate": vectors,
        "vectors": vectors,
        "output_vectors": outv,
        "vocab": dict((w, counts[w]) for w in vocab),
        "loss_curve": curve,
        "loss": curve[-1] if curve else float("nan"),
        "similarity": similarity,
        "most_similar": most_similar,
        "size": size,
        "window": window,
        "architecture": architecture,
        "method": "word2vec (Mikolov et al. 2013, sections 3.1-3.2)",
    })


def _scores(O, h, V, size):
    return [sum(O[k][d] * h[d] for d in range(size)) for k in range(V)]


def _sg_step(W, O, c, j, size, V, lr):
    """Skip-gram: input is the centre word, target a context word."""
    h = W[c]
    p = _softmax(_scores(O, h, V, size))
    loss = -math.log(max(p[j], 1e-300))
    # dL/dscore_k = p_k - [k == j]
    gh = [0.0] * size
    for k in range(V):
        e = p[k] - (1.0 if k == j else 0.0)
        if e == 0.0:
            continue
        for d in range(size):
            gh[d] += e * O[k][d]
        for d in range(size):
            O[k][d] -= lr * e * h[d]
    for d in range(size):
        W[c][d] -= lr * gh[d]
    return loss


def _cbow_step(W, O, ctx, c, size, V, lr):
    """CBOW: the projection is the MEAN of the context vectors."""
    n = float(len(ctx))
    h = [0.0] * size
    for i in ctx:
        for d in range(size):
            h[d] += W[i][d]
    for d in range(size):
        h[d] /= n
    p = _softmax(_scores(O, h, V, size))
    loss = -math.log(max(p[c], 1e-300))
    gh = [0.0] * size
    for k in range(V):
        e = p[k] - (1.0 if k == c else 0.0)
        if e == 0.0:
            continue
        for d in range(size):
            gh[d] += e * O[k][d]
        for d in range(size):
            O[k][d] -= lr * e * h[d]
    # the averaging shares the gradient equally across the context
    for i in ctx:
        for d in range(size):
            W[i][d] -= lr * gh[d] / n
    return loss


def _cos(a, b):
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def analogy(vectors, a, b, c, topn=1):
    r"""Section 4's offset query: ``b - a + c``.

    ``analogy(v, "Man", "King", "Woman")`` asks "man is to king as
    woman is to what". The three question words are excluded from the
    answer, as the paper's evaluation does.
    """
    for w in (a, b, c):
        if w not in vectors:
            raise KeyError("analogy: %r is not in the vocabulary" % (w,))
    size = len(vectors[a])
    target = [vectors[b][d] - vectors[a][d] + vectors[c][d]
              for d in range(size)]
    sims = [(w, _cos(target, v)) for w, v in vectors.items()
            if w not in (a, b, c)]
    sims.sort(key=lambda p: -p[1])
    return sims[:topn]


def cheatsheet():
    return ("wrd2v: log-linear word vectors (Mikolov 2013). CBOW "
            "predicts the centre word from AVERAGED context vectors "
            "(sec 3.1, Q = N*D + D*log2 V); skip-gram predicts context "
            "from the centre word (sec 3.2, Q = C*(D + D*log2 V)) with "
            "a DYNAMIC window R ~ Unif{1..C}, so distance d is used "
            "with probability (C-d+1)/C. Negative sampling is the "
            "FOLLOW-UP paper, not this one. analogy() is the "
            "b - a + c offset query of sec 4.")


# compact alias per ledger/NAMING.md
word2vec = wrd2v
