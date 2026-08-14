# morie.fn -- function file (rootcoder007/morie)
r"""DSSM: word hashing, then a semantic space trained on clicks.

Latent semantic models for web search had two problems: they were
trained on **objective functions unrelated to retrieval** (word
co-occurrence, reconstruction), and the vocabulary of a real query
stream is far too large for a neural network's input layer.

**Word hashing solves the vocabulary problem with letter n-grams.** A
word is bracketed (``#good#``) and represented by the multiset of its
letter trigrams. A 500K-word vocabulary collapses to about 30K
trigrams -- and the collapse is *not* lossy in the way a hash table is:
two different words share a representation only if they share every
trigram, so ``collision_rate`` measures the real cost, which the paper
finds negligible. The side benefit is that an out-of-vocabulary or
misspelled word still has a representation, because its trigrams exist
even if the word never appeared in training.

**The objective is clickthrough, which is the actual retrieval
signal.** Query and document are projected into a common semantic
space, scored by **cosine similarity**, and the model maximises the
conditional likelihood of the **clicked** document under a softmax
over the clicked one plus randomly sampled unclicked ones. Training on
what users clicked is the point of departure from the earlier models.

**The smoothing factor is not decoration.** The cosine similarity is
scaled by :math:`\gamma` inside the softmax; it sets how sharply the
posterior concentrates, and at :math:`\gamma \to 0` every document is
equally likely no matter what the model learned.

References
----------
Huang, P.-S., He, X., Gao, J., Deng, L., Acero, A. & Heck, L. (2013)
"Learning Deep Structured Semantic Models for Web Search using
Clickthrough Data", *Proceedings of the 22nd ACM International
Conference on Information and Knowledge Management (CIKM '13)*,
2333-2338, doi:10.1145/2505515.2505665. Sec. 3: the deep structured
semantic model projecting queries and documents into a common
low-dimensional semantic space where relevance is computed by cosine
similarity; WORD HASHING based on letter n-grams, which reduces the
dimensionality of the bag-of-words term vectors (a 500K vocabulary to
roughly 30K letter trigrams) with a very low collision rate and gives
representations to out-of-vocabulary and misspelled words; the
training objective maximising the conditional likelihood of the
CLICKED documents given the query under a softmax over the clicked
document and randomly sampled unclicked documents, with a smoothing
factor gamma in the softmax; and the criticism of earlier latent
semantic models as trained on objective functions loosely related to
the retrieval task.

Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. &
Harshman, R. (1990) "Indexing by latent semantic analysis", *Journal
of the American Society for Information Science* 41(6), 391-407. The
unsupervised alternative being displaced.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["letter_ngrams", "word_hash", "collision_rate",
           "cosine_similarity", "click_posterior"]

_EPS = 1e-12


def letter_ngrams(word, n=3, boundary="#"):
    r"""The trigrams of ``#word#``.

    Bracketing matters: it distinguishes a prefix from the same
    letters inside a word.
    """
    w = boundary + str(word).lower() + boundary
    m = int(n)
    if m < 1:
        raise ValueError("dssm: n must be at least 1")
    if len(w) < m:
        return [w]
    return [w[i:i + m] for i in range(len(w) - m + 1)]


def word_hash(words, n=3, vocabulary=None):
    r"""A bag of letter n-grams instead of a bag of words.

    The input layer's width becomes the number of n-grams, not the
    size of the vocabulary -- and an unseen word still has a vector.
    """
    W = [str(v) for v in words]
    grams = {}
    for w in W:
        for g in letter_ngrams(w, n):
            grams[g] = grams.get(g, 0) + 1
    keys = sorted(vocabulary) if vocabulary is not None \
        else sorted(grams)
    idx = {g: i for i, g in enumerate(keys)}
    vec = [0.0] * len(keys)
    unseen = 0
    for g, c in grams.items():
        if g in idx:
            vec[idx[g]] += c
        else:
            unseen += 1
    return {"vector": vec, "dimension": len(keys),
            "ngrams": grams, "unseen_ngrams": unseen,
            "note": "an out-of-vocabulary or misspelled word still "
                    "has trigrams, so it still has a representation"}


def collision_rate(vocabulary, n=3):
    r"""Two words collide only if they share EVERY n-gram.

    Not a hash function's arbitrary collision -- the cost is real but
    measurable, and the paper finds it negligible.
    """
    V = [str(v) for v in vocabulary]
    if not V:
        raise ValueError("dssm: the vocabulary is empty")
    buckets = {}
    for w in V:
        key = tuple(sorted(letter_ngrams(w, n)))
        buckets.setdefault(key, []).append(w)
    collided = [ws for ws in buckets.values() if len(ws) > 1]
    n_col = sum(len(ws) for ws in collided)
    grams = set()
    for w in V:
        grams.update(letter_ngrams(w, n))
    return {"vocabulary": len(V), "ngram_dimension": len(grams),
            "reduction": len(V) / float(len(grams)),
            "collisions": n_col,
            "collision_rate": n_col / float(len(V)),
            "colliding_groups": [sorted(ws) for ws in collided],
            "note": "the input layer shrinks from |V| to |n-grams|"}


def cosine_similarity(query_vector, doc_vector):
    r"""Relevance in the common semantic space."""
    q = [float(v) for v in k.vec(query_vector)]
    d = [float(v) for v in k.vec(doc_vector)]
    if len(q) != len(d):
        raise ValueError("dssm: the query and document vectors "
                         "differ in width")
    nq = math.sqrt(sum(v * v for v in q))
    nd = math.sqrt(sum(v * v for v in d))
    if nq <= _EPS or nd <= _EPS:
        raise ValueError("dssm: a zero vector has no direction, so "
                         "cosine similarity is undefined")
    return sum(q[i] * d[i] for i in range(len(q))) / (nq * nd)


def click_posterior(query_vector, clicked_vector,
                    unclicked_vectors, gamma=10.0):
    r"""Softmax over the clicked document and sampled unclicked ones.

    :math:`\gamma` sets how sharply the posterior concentrates; at
    :math:`\gamma\to 0` every document is equally likely whatever the
    model learned.
    """
    g = float(gamma)
    if g <= 0.0:
        raise ValueError("dssm: the smoothing factor must be "
                         "positive")
    sims = [cosine_similarity(query_vector, clicked_vector)]
    for d in unclicked_vectors:
        sims.append(cosine_similarity(query_vector, d))
    sc = [g * v for v in sims]
    m = max(sc)
    e = [math.exp(v - m) for v in sc]
    z = sum(e)
    p = [v / z for v in e]
    return RichResult(payload={
        "estimate": p[0], "posterior_clicked": p[0],
        "posterior": p, "similarities": sims, "gamma": g,
        "loss": -math.log(max(p[0], _EPS)),
        "n_negatives": len(unclicked_vectors),
        "method": "clickthrough-trained semantic model; Huang et al. "
                  "(2013)",
        "note": "trained on what users CLICKED, not on word "
                "co-occurrence",
    })


def cheatsheet():
    return ("dssm: earlier latent semantic models were trained on "
            "objectives only loosely related to RETRIEVAL, and a real "
            "query vocabulary is too large for an input layer. WORD "
            "HASHING fixes the second: represent a word by the letter "
            "trigrams of #word#, so 500K words become ~30K trigrams, "
            "two words collide only if they share EVERY trigram, and "
            "an out-of-vocabulary or misspelled word still has a "
            "vector. The objective fixes the first: project query and "
            "document into one semantic space, score by COSINE, and "
            "maximise the likelihood of the CLICKED document under a "
            "softmax with smoothing factor gamma over sampled "
            "unclicked ones.")


# compact alias per ledger/NAMING.md
deepsemanticmodel = click_posterior

# public names resolved by fn/_lazy_map.json
dssm = click_posterior
