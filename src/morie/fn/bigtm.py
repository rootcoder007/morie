# morie.fn -- function file (rootcoder007/morie)
r"""The bigram topic model: topics that know about word order.

Two traditions had not been combined. n-gram models predict each word
from the words before it; latent-topic models infer topics from word
co-occurrence with the order thrown away. This model extends latent
Dirichlet allocation by giving each topic a *set* of :math:`W`
distributions -- one per preceding word -- so word generation is
conditioned on both the previous word and the current topic.

**Why order helps topic inference, not just prediction.** "The
department chair couches offers" and "the chair department offers
couches" have identical unigram statistics and are about different
things. Knowing that "chair" was preceded by "department" makes it far
more likely to have come from a topic about university administration.
A bag-of-words model cannot see that distinction at all -- which is
exactly what the anchor tests.

**And a practical consequence.** Topics inferred by LDA are dominated
by function words -- "in", "that", "of", "for" -- unless those are
stripped from the corpus beforehand. That is fine for retrieval and
wrong for language modelling, where function words must also be
predicted well. Conditioning on the previous word lets the function
words be explained by the bigram structure instead of the topics.

**The smoothing that makes bigram statistics usable.** Only a small
fraction of word pairs are ever observed, so :math:`f_{i|j} =
N_{i|j}/N_j` has far too much variance to use alone. The classical
answer interpolates with the marginal (eq. 1). MacKay and Peto's
hierarchical Dirichlet model reaches the same shape from Bayes:

.. math:: P(i \mid j, w, \beta m) = \frac{N_{i|j} + \beta m_i}
                                         {N_j + \beta}
          = \lambda_j m_i + (1-\lambda_j) f_{i|j}, \qquad
          \lambda_j = \frac{\beta}{N_j + \beta},

(eqs. 5-7) so the hyperparameter :math:`m_i` takes the role the
marginal frequency played -- and, under empirical Bayes, the optimal
:math:`m_i` turns out to relate to the *number of contexts* in which
word :math:`i` appeared, not its raw count.

**The new model.** Word generation is
:math:`P(w_t = i \mid w_{t-1} = j, z_t = k) \equiv \phi_{i|j,k}`, with
:math:`WT(W-1)` free parameters, and topics are drawn per document as
in LDA. That leaves a genuine choice of how to share the Dirichlet
prior, and both options are implemented:

* **prior 1** (eq. 21) -- one hyperparameter vector :math:`\beta m`
  shared by every context :math:`j,k`, so :math:`N_{i|j,k}/N_{j,k}` is
  smoothed by the same :math:`m_i` regardless of topic;
* **prior 2** (eq. 22) -- one vector :math:`\beta_k m_k` per topic, so
  the smoothing varies with the topic. Information is shared only
  between contexts with the same topic, which is the intuitively
  appealing reading: learning about :math:`j,k` should say something
  about :math:`j',k` and nothing about another topic.

Inference is Gibbs sampling over :math:`z` (eqs. 28-29), inside a Gibbs
EM loop that optimises the hyperparameters by maximising the evidence.

**Two printed equations disagree with their own algebra**, and this
module follows the algebra:

* Eq. (15) reads :math:`\lambda_k f_{i|k} + (1-\lambda_k)m_i` with
  :math:`\lambda_k = \beta/(N_k+\beta)`. Expanding eq. (13) directly
  gives the *opposite* assignment -- weight :math:`N_k/(N_k+\beta)` on
  :math:`f_{i|k}` and :math:`\beta/(N_k+\beta)` on :math:`m_i` -- which
  is also what eq. (6) says for the analogous case. Eq. (13) is
  implemented; the anchor shows the two disagree.
* Eq. (28) divides by :math:`\{N_k\}_{-t} + \beta`, but the predictive
  distribution it comes from (eq. 25) divides by :math:`N_{j,k}+\beta`,
  the count for the *context*, and eq. (29) uses
  :math:`\{N_{w_{t-1},k}\}_{-t} + \beta_k`. The context denominator is
  used here.

References
----------
Wallach, H. M. (2006) "Topic Modeling: Beyond Bag-of-Words",
*Proceedings of the 23rd International Conference on Machine Learning
(ICML 2006)*, 977-984, doi:10.1145/1143844.1143967. Sec. 1 (the two
traditions; the "department chair" example; the observation that LDA
topics are dominated by function words unless these are removed, which
is inappropriate for language modelling). Sec. 2.1 (bigram smoothing,
eq. (1); MacKay and Peto's hierarchical Dirichlet language model,
eqs. (2)-(7); the empirical-Bayes result that the optimal m relates to
the number of contexts a word appears in). Sec. 2.2 (LDA, eqs.
(9)-(18)). Sec. 3 (the bigram topic model, eqs. (19)-(26), prior 1 and
prior 2 and the argument for each). Sec. 4 (Gibbs EM and the
conditional posteriors, eqs. (28)-(29)).

MacKay, D. J. C. & Peto, L. C. B. (1995) "A hierarchical Dirichlet
language model", *Natural Language Engineering* 1(3), 289-308,
doi:10.1017/S1351324900000218. The hierarchical Dirichlet bigram
model whose prior structure is imported here.

Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) "Latent Dirichlet
Allocation", *Journal of Machine Learning Research* 3, 993-1022. The
model being extended; implemented in :mod:`lda`.

Griffiths, T. L. & Steyvers, M. (2004) "Finding scientific topics",
*Proceedings of the National Academy of Sciences* 101(suppl. 1),
5228-5235, doi:10.1073/pnas.0307752101. The Gibbs sampler used to
approximate the intractable sum over z.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dirichlet_predictive", "lda_predictive",
           "bigram_topic_predictive", "gibbs_bigram_topic",
           "log_evidence"]

_EPS = 1e-300
_PRIORS = (1, 2)


def dirichlet_predictive(N_ij, N_j, beta, m):
    r"""Eq. (5): :math:`(N_{i|j} + \beta m_i)/(N_j + \beta)`.

    Also returned in the eq. (6)-(7) form, which makes the
    interpolation with the classical estimator explicit.
    """
    n = [float(v) for v in k.vec(N_ij)]
    mm = [float(v) for v in k.vec(m)]
    if len(n) != len(mm):
        raise ValueError("bigtm: %d counts for %d prior weights"
                         % (len(n), len(mm)))
    if abs(sum(mm) - 1.0) > 1e-9:
        raise ValueError("bigtm: m must sum to 1, got %.9f" % sum(mm))
    b, Nj = float(beta), float(N_j)
    if b <= 0.0:
        raise ValueError("bigtm: beta must be positive")
    lam = b / (Nj + b)
    f = [v / Nj if Nj > 0 else 0.0 for v in n]
    return {"predictive": [(n[i] + b * mm[i]) / (Nj + b)
                           for i in range(len(n))],
            "lambda": lam, "f": f,
            "interpolated": [lam * mm[i] + (1.0 - lam) * f[i]
                             for i in range(len(n))],
            "note": "eq. (6): lambda_j m_i + (1 - lambda_j) f_{i|j}, "
                    "so m_i plays the role of the marginal frequency"}


def lda_predictive(N_ik, N_k, beta, m):
    r"""Eq. (13): the LDA topic-word predictive distribution.

    Returned alongside the printed eq. (15) form so the two can be
    compared -- they do not agree, and eq. (13) is the one that follows
    from the counts.
    """
    n = [float(v) for v in k.vec(N_ik)]
    mm = [float(v) for v in k.vec(m)]
    b, Nk = float(beta), float(N_k)
    if b <= 0.0:
        raise ValueError("bigtm: beta must be positive")
    lam = b / (Nk + b)
    f = [v / Nk if Nk > 0 else 0.0 for v in n]
    return {"predictive": [(n[i] + b * mm[i]) / (Nk + b)
                           for i in range(len(n))],
            "lambda": lam,
            "eq15_as_printed": [lam * f[i] + (1.0 - lam) * mm[i]
                                for i in range(len(n))],
            "note": "eq. (15) as printed puts weight lambda_k on "
                    "f_{i|k}; expanding eq. (13) puts it on m_i, as "
                    "eq. (6) does. Eq. (13) is used."}


def bigram_topic_predictive(N_ijk, N_jk, beta, m, prior=1):
    r"""Eqs. (25)-(26): the word predictive given previous word and
    topic.

    ``prior=1`` shares one :math:`\beta m` across all contexts;
    ``prior=2`` takes topic-specific :math:`\beta_k m_k`, in which case
    ``beta`` and ``m`` are the ones for that topic.
    """
    if int(prior) not in _PRIORS:
        raise ValueError("bigtm: prior must be 1 or 2, got %r"
                         % (prior,))
    n = [float(v) for v in k.vec(N_ijk)]
    mm = [float(v) for v in k.vec(m)]
    b = float(beta)
    return {"predictive": [(n[i] + b * mm[i]) / (float(N_jk) + b)
                           for i in range(len(n))],
            "prior": int(prior),
            "smoothed_by": ("m_i, the same for every context"
                            if int(prior) == 1
                            else "m_{i|k}, which varies with the "
                                 "topic")}


def _counts(docs, T, V, z):
    N_ijk = {}
    N_jk = {}
    N_kd = [[0.0] * T for _ in docs]
    N_d = [0.0] * len(docs)
    for d, doc in enumerate(docs):
        for t in range(1, len(doc)):
            i, j, kk = doc[t], doc[t - 1], z[d][t]
            N_ijk[(i, j, kk)] = N_ijk.get((i, j, kk), 0.0) + 1.0
            N_jk[(j, kk)] = N_jk.get((j, kk), 0.0) + 1.0
            N_kd[d][kk] += 1.0
            N_d[d] += 1.0
    return N_ijk, N_jk, N_kd, N_d


def gibbs_bigram_topic(docs, T, V, alpha=0.5, beta=0.5, m=None,
                       n=None, prior=1, iters=200, seed=0, burn=50):
    r"""Eq. (28)/(29): Gibbs sampling for the topic assignments.

    The first token of each document has no preceding word, so it is
    not assigned; sampling runs over positions :math:`t \ge 2`.
    """
    if int(prior) not in _PRIORS:
        raise ValueError("bigtm: prior must be 1 or 2, got %r"
                         % (prior,))
    D = [[int(v) for v in d] for d in docs]
    if not D:
        raise ValueError("bigtm: no documents given")
    Tn, Vn = int(T), int(V)
    if Tn < 1 or Vn < 1:
        raise ValueError("bigtm: T and V must be at least 1")
    if any(v < 0 or v >= Vn for d in D for v in d):
        raise ValueError("bigtm: a word index is outside the "
                         "vocabulary of %d" % Vn)
    mm = [1.0 / Vn] * Vn if m is None else [float(v) for v in k.vec(m)]
    nn = [1.0 / Tn] * Tn if n is None else [float(v) for v in k.vec(n)]
    if abs(sum(mm) - 1.0) > 1e-9 or abs(sum(nn) - 1.0) > 1e-9:
        raise ValueError("bigtm: m and n must each sum to 1")
    a, b = float(alpha), float(beta)
    rng = np.random.default_rng(seed)
    z = [[int(rng.uniform() * Tn) % Tn for _ in d] for d in D]
    N_ijk, N_jk, N_kd, N_d = _counts(D, Tn, Vn, z)
    acc = [[[0.0] * Tn for _ in d] for d in D]
    kept = 0
    for it in range(int(iters)):
        for d in range(len(D)):
            for t in range(1, len(D[d])):
                i, j = D[d][t], D[d][t - 1]
                old = z[d][t]
                N_ijk[(i, j, old)] -= 1.0
                N_jk[(j, old)] -= 1.0
                N_kd[d][old] -= 1.0
                p = []
                for kk in range(Tn):
                    w = ((N_ijk.get((i, j, kk), 0.0) + b * mm[i])
                         / (N_jk.get((j, kk), 0.0) + b))
                    p.append(w * (N_kd[d][kk] + a * nn[kk]))
                s = sum(p)
                u = float(rng.uniform()) * s
                new, c = Tn - 1, 0.0
                for kk in range(Tn):
                    c += p[kk]
                    if u <= c:
                        new = kk
                        break
                z[d][t] = new
                N_ijk[(i, j, new)] = N_ijk.get((i, j, new), 0.0) + 1.0
                N_jk[(j, new)] = N_jk.get((j, new), 0.0) + 1.0
                N_kd[d][new] += 1.0
        if it >= int(burn):
            kept += 1
            for d in range(len(D)):
                for t in range(1, len(D[d])):
                    acc[d][t][z[d][t]] += 1.0
    post = [[[v / kept for v in acc[d][t]] if kept else acc[d][t]
             for t in range(len(D[d]))] for d in range(len(D))]
    theta = [[(N_kd[d][kk] + a * nn[kk]) / (N_d[d] + a)
              for kk in range(Tn)] for d in range(len(D))]
    return RichResult(payload={
        "estimate": z, "z": z, "topic_posterior": post,
        "theta": theta, "N_ijk": N_ijk, "N_jk": N_jk,
        "T": Tn, "V": Vn, "prior": int(prior),
        "iterations": int(iters), "burn_in": int(burn),
        "samples_kept": kept,
        "method": "Gibbs sampling for the bigram topic model; "
                  "Wallach (2006) eqs. (28)-(29)",
        "caveat": "eq. (28) as printed divides by {N_k}_-t + beta; "
                  "the context count N_{j,k} + beta is used, "
                  "following eqs. (25) and (29)",
    })


def log_evidence(docs, T, V, z, alpha=0.5, beta=0.5, m=None, n=None):
    r"""Eq. (23): :math:`\log P(w, z \mid \alpha n, \beta m)` for one
    :math:`z`.
    """
    D = [[int(v) for v in d] for d in docs]
    Tn, Vn = int(T), int(V)
    mm = [1.0 / Vn] * Vn if m is None else [float(v) for v in k.vec(m)]
    nn = [1.0 / Tn] * Tn if n is None else [float(v) for v in k.vec(n)]
    a, b = float(alpha), float(beta)
    N_ijk, N_jk, N_kd, N_d = _counts(D, Tn, Vn, z)
    tot = 0.0
    for (j, kk), njk in N_jk.items():
        tot += k.lgamma(b) - k.lgamma(njk + b)
        for i in range(Vn):
            c = N_ijk.get((i, j, kk), 0.0)
            tot += k.lgamma(c + b * mm[i]) - k.lgamma(b * mm[i])
    for d in range(len(D)):
        tot += k.lgamma(a) - k.lgamma(N_d[d] + a)
        for kk in range(Tn):
            tot += k.lgamma(N_kd[d][kk] + a * nn[kk]) \
                - k.lgamma(a * nn[kk])
    return tot


def cheatsheet():
    return ("bigtm: LDA where each topic holds W distributions, one "
            "per PRECEDING word, so P(w_t = i | w_{t-1} = j, z_t = k). "
            "Word order helps topic inference -- 'department chair' vs "
            "'chair department' have identical unigram statistics -- "
            "and it stops function words dominating the topics. Two "
            "priors, both implemented: ONE beta*m shared across all "
            "contexts (prior 1), or one per TOPIC (prior 2), which "
            "shares information only within a topic. Gibbs EM. Note "
            "eqs. (15) and (28) as printed contradict eqs. (13) and "
            "(25); the algebra is followed, not the typography.")


# compact alias per ledger/NAMING.md
bigramtopicmodel = gibbs_bigram_topic
