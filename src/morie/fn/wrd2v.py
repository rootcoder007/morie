r"""word2vec: the CBOW and continuous skip-gram architectures.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013) "Efficient
Estimation of Word Representations in Vector Space",
arXiv:1301.3781 -- the CBOW and skip-gram architectures.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013)
"Distributed Representations of Words and Phrases and their
Compositionality", *NeurIPS*, arXiv:1310.4546 -- negative sampling and
subsampling of frequent words, which are extensions of skip-gram
published separately and are marked as such below.

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

The output layer is the full softmax by default. The 2013a complexity
terms are written with :math:`\log_2 V`, reflecting the hierarchical
softmax used at scale; :func:`training_complexity` reports :math:`Q`
for either.

**Negative sampling** (``loss="neg"``) comes from the *second* paper,
Mikolov et al. (2013b) section 2.2, and replaces every
:math:`\log P(w_O \mid w_I)` term in the skip-gram objective with
(their eq. 4)

.. math:: \log \sigma\big(v'^{\top}_{w_O} v_{w_I}\big)
          + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}
          \Big[\log \sigma\big(-v'^{\top}_{w_i} v_{w_I}\big)\Big].

It is a simplification of Noise Contrastive Estimation: the task is
just to tell the true target from :math:`k` draws of a noise
distribution by logistic regression. NCE needs the numerical noise
probabilities; NEG needs only samples. The paper reports
:math:`k = 5\text{-}20` for small datasets and :math:`2\text{-}5` for
large ones, and that the noise distribution which "outperformed
significantly the unigram and the uniform distributions ... on every
task we tried" is the unigram raised to the 3/4 power,

.. math:: P_n(w) = U(w)^{3/4} / Z.

That exponent is not folklore -- it is the paper's empirical finding,
and ``noise_power`` exposes it rather than hard-coding it.

**Subsampling of frequent words** (2013b section 2.3, their eq. 5)
discards each occurrence of :math:`w_i` with probability

.. math:: P(w_i) = 1 - \sqrt{t / f(w_i)},

with :math:`f(w_i)` the word's frequency and :math:`t` a threshold
(they use around :math:`10^{-5}`). Words rarer than :math:`t` are
never discarded, since the formula goes negative and is clamped at
zero. Set ``subsample=None`` to switch it off.

What the vectors are for is section 4's observation that they encode
regularities as *offsets*: ``vector("King") - vector("Man") +
vector("Woman")`` lands near ``vector("Queen")``.
:func:`analogy` performs that query, excluding the three question
words from the answer as the paper's evaluation does.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wrd2v", "word2vec", "analogy", "training_complexity",
           "noise_distribution", "subsample_probability"]

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


def noise_distribution(counts, power=0.75):
    r""":math:`P_n(w) = U(w)^{3/4} / Z` of Mikolov et al. (2013b) 2.2.

    ``counts`` maps word to frequency. The 3/4 exponent is the
    paper's empirical choice; ``power=1`` gives the plain unigram and
    ``power=0`` the uniform distribution, both of which it reports as
    significantly worse.
    """
    ws = sorted(counts, key=repr)
    raw = [float(counts[w]) ** float(power) for w in ws]
    z = sum(raw)
    if z <= 0.0:
        raise ValueError("wrd2v: noise distribution has no mass")
    return dict((ws[i], raw[i] / z) for i in range(len(ws)))


def subsample_probability(counts, t=1e-5):
    r"""Discard probability :math:`1 - \sqrt{t / f(w)}` (2013b eq. 5).

    ``f(w)`` is the relative frequency. Clamped at zero, so a word
    rarer than :math:`t` is never discarded.
    """
    total = float(sum(counts.values()))
    if total <= 0.0:
        raise ValueError("wrd2v: empty counts")
    t = float(t)
    if t <= 0.0:
        raise ValueError("wrd2v: t must be > 0")
    out = {}
    for w, c in counts.items():
        f = c / total
        out[w] = max(0.0, 1.0 - math.sqrt(t / f))
    return out


def wrd2v(corpus, size=16, window=5, architecture="skip-gram", lr=0.05,
          epochs=20, min_count=1, dynamic_window=True, loss="softmax",
          negative=5, noise_power=0.75, subsample=None, seed=0):
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
    loss : {"softmax", "neg"}
        Full softmax (2013a) or negative sampling (2013b eq. 4).
        Skip-gram only -- eq. 4 is defined as a replacement for the
        terms of the *skip-gram* objective.
    negative : int
        :math:`k`, the number of noise samples per positive. The paper
        suggests 5-20 for small corpora, 2-5 for large ones.
    noise_power : float
        Exponent of :math:`P_n(w) \propto U(w)^{\text{power}}`; the
        paper's 3/4 by default.
    subsample : float, optional
        The :math:`t` of 2013b eq. 5. ``None`` disables subsampling.
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
    if loss not in ("softmax", "neg"):
        raise ValueError("wrd2v: loss must be 'softmax' or 'neg', got %r"
                         % (loss,))
    if loss == "neg" and architecture != "skip-gram":
        raise ValueError("wrd2v: negative sampling is defined in Mikolov "
                         "et al. (2013b) eq. 4 as a replacement for the "
                         "terms of the SKIP-GRAM objective; use "
                         "architecture='skip-gram' or loss='softmax'")
    negative = int(negative)
    if loss == "neg" and negative < 1:
        raise ValueError("wrd2v: negative must be >= 1")
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

    # 2013b eq. 5: discard frequent words before training.
    if subsample is not None:
        keep_drop = subsample_probability(
            dict((w, counts[w]) for w in vocab), subsample)
        srng = np.random.default_rng(seed + 7)
        sents = [[w for w in s if srng.random() >= keep_drop[w]]
                 for s in sents]

    rng = np.random.default_rng(seed)
    # 2013b 2.2: the noise distribution, as a cumulative table.
    noise = noise_distribution(dict((w, counts[w]) for w in vocab),
                               noise_power)
    cum = []
    acc = 0.0
    for w in vocab:
        acc += noise[w]
        cum.append(acc)

    def draw_noise():
        u = rng.random() * cum[-1]
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

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
                elif loss == "neg":
                    for j in ctx:
                        total += _neg_step(W, O, c, j, size, lr,
                                           negative, draw_noise)
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
        "final_loss": curve[-1] if curve else float("nan"),
        "similarity": similarity,
        "most_similar": most_similar,
        "size": size,
        "window": window,
        "architecture": architecture,
        "loss": loss,
        "negative": negative if loss == "neg" else 0,
        "noise": noise,
        "method": "word2vec (Mikolov et al. 2013a secs 3.1-3.2"
                  + ("; 2013b eq. 4 negative sampling)"
                     if loss == "neg" else ")"),
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


def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _neg_step(W, O, c, j, size, lr, k, draw_noise):
    r"""Mikolov et al. (2013b) eq. 4, one positive and k noise draws.

    The loss minimised is the negative of eq. 4:
        -log sigma(o_j . w_c) - sum_i log sigma(-o_{n_i} . w_c).
    """
    h = W[c]
    targets = [(j, 1.0)]
    for _ in range(k):
        n = draw_noise()
        targets.append((n, 0.0))
    gh = [0.0] * size
    loss = 0.0
    for idx_t, label in targets:
        z = sum(O[idx_t][d] * h[d] for d in range(size))
        p = _sigmoid(z)
        loss -= math.log(max(p if label == 1.0 else 1.0 - p, 1e-300))
        e = p - label
        for d in range(size):
            gh[d] += e * O[idx_t][d]
        for d in range(size):
            O[idx_t][d] -= lr * e * h[d]
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
    return ("wrd2v: log-linear word vectors (Mikolov 2013a). CBOW "
            "predicts the centre word from AVERAGED context vectors "
            "(sec 3.1, Q = N*D + D*log2 V); skip-gram predicts context "
            "from the centre word (sec 3.2, Q = C*(D + D*log2 V)) with "
            "a DYNAMIC window R ~ Unif{1..C}, so distance d is used "
            "with probability (C-d+1)/C. loss='neg' is negative "
            "sampling from the FOLLOW-UP paper (2013b eq. 4) with "
            "Pn(w) = U(w)^0.75/Z, plus eq. 5 subsampling. analogy() "
            "is the b - a + c offset query of 2013a sec 4.")


# compact alias per ledger/NAMING.md
word2vec = wrd2v
