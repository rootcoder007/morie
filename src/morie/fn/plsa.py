# morie.fn -- function file (rootcoder007/morie)
r"""Probabilistic Latent Semantic Analysis by EM.

LSA maps documents and terms into a low-dimensional latent space by
truncated SVD of the term-document matrix. It works, and its
theoretical foundation is, in Hofmann's words, unsatisfactory and
incomplete: the objective is an L2 matrix approximation, which is not
a statement about counts, and the resulting coordinates can be
negative.

**The aspect model gives it a generative story.** Associate an
unobserved class :math:`z \in \{z_1,\dots,z_K\}` with each observation.
The joint over documents and words is then, in the asymmetric
parameterisation (eq. 1),

.. math:: P(d, w) = P(d)P(w \mid d), \qquad
          P(w \mid d) = \sum_{z} P(w \mid z)P(z \mid d),

and, equivalently, in the symmetric one (eq. 2),

.. math:: P(d, w) = \sum_{z} P(z)P(d \mid z)P(w \mid z).

The two are the same model written from different sides -- the second
is symmetric in documents and words, which makes the conditional
independence explicit: :math:`d` and :math:`w` are independent **given
z**. Because :math:`|z|` is smaller than the number of documents or
words, :math:`z` is a bottleneck, and that bottleneck is what forces
the model to find structure.

**EM, in closed form.** The E step (eq. 3) is Bayes' rule over the
latent class,

.. math:: P(z \mid d, w) = \frac{P(z)P(d\mid z)P(w \mid z)}
          {\sum_{z'} P(z')P(d\mid z')P(w \mid z')},

and the M step (eqs. 4-6) is expected-count normalisation,

.. math:: P(w \mid z) &\propto \sum_d n(d,w)P(z\mid d,w),\\
          P(d \mid z) &\propto \sum_w n(d,w)P(z\mid d,w),\\
          P(z) &\propto \sum_d\sum_w n(d,w)P(z\mid d,w).

**What it fixes and what it does not.** Against LSA it gains a proper
likelihood -- maximising it minimises the KL divergence between the
empirical and modelled distributions -- and non-negative, interpretable
parameters. What it does not gain is a generative story for *unseen*
documents: :math:`P(z \mid d)` is a parameter fitted per training
document, so the number of parameters grows with the corpus and a new
document requires re-running EM. Putting a Dirichlet prior on
:math:`P(z\mid d)` is exactly what :mod:`lda` does, and that is the
gap it closes.

References
----------
Hofmann, T. (1999) "Probabilistic Latent Semantic Analysis",
*Proceedings of the Fifteenth Conference on Uncertainty in Artificial
Intelligence (UAI 1999)*, 289-296, arXiv:1301.6705. Sec. 2 (LSA by
SVD and the assessment of its theoretical foundation), Sec. 3 (the
aspect model, eq. (1) asymmetric and eq. (2) symmetric
parameterisations, the conditional independence of d and w given z,
and z as a bottleneck), and Sec. 3.2 (EM: eq. (3) for the E step and
eqs. (4)-(6) for the M step; maximum likelihood as minimisation of
the cross entropy or KL divergence against the empirical
distribution).

NOTE: the text layer of the local PDF is garbled across the equation
region; the equations above were recovered by OCR (ocrpg.sh, 300 dpi
+ tesseract) rather than by pdftotext.

Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. &
Harshman, R. (1990) "Indexing by latent semantic analysis", *Journal
of the American Society for Information Science* 41(6), 391-407. The
LSA method this gives a probabilistic footing to.

Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) "Latent Dirichlet
Allocation", *Journal of Machine Learning Research* 3, 993-1022. The
Dirichlet prior that removes the per-document parameter growth;
implemented in :mod:`lda`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["e_step", "m_step", "log_likelihood", "fit_plsa",
           "joint_probability", "perplexity"]

_EPS = 1e-300


def _check(n_dw):
    N = [[float(v) for v in r] for r in k.mat(n_dw)]
    if not N:
        raise ValueError("plsa: the count matrix is empty")
    if any(len(r) != len(N[0]) for r in N):
        raise ValueError("plsa: the count matrix is ragged")
    if any(v < 0.0 for r in N for v in r):
        raise ValueError("plsa: counts must be non-negative")
    if sum(sum(r) for r in N) <= 0.0:
        raise ValueError("plsa: the corpus is empty")
    return N, len(N), len(N[0])


def e_step(n_dw, Pz, Pd_z, Pw_z):
    r"""Eq. (3): the posterior over the latent class."""
    N, D, V = _check(n_dw)
    K = len(Pz)
    post = [[[0.0] * K for _ in range(V)] for _ in range(D)]
    for d in range(D):
        for w in range(V):
            if N[d][w] <= 0.0:
                continue
            num = [Pz[z] * Pd_z[z][d] * Pw_z[z][w] for z in range(K)]
            s = sum(num)
            if s <= _EPS:
                post[d][w] = [1.0 / K] * K
            else:
                post[d][w] = [v / s for v in num]
    return post


def m_step(n_dw, post, K):
    r"""Eqs. (4)-(6): expected counts, normalised."""
    N, D, V = _check(n_dw)
    Pw_z = [[_EPS] * V for _ in range(K)]
    Pd_z = [[_EPS] * D for _ in range(K)]
    Pz = [_EPS] * K
    for d in range(D):
        for w in range(V):
            c = N[d][w]
            if c <= 0.0:
                continue
            for z in range(K):
                r = c * post[d][w][z]
                Pw_z[z][w] += r
                Pd_z[z][d] += r
                Pz[z] += r
    for z in range(K):
        a = sum(Pw_z[z])
        Pw_z[z] = [v / a for v in Pw_z[z]]
        b = sum(Pd_z[z])
        Pd_z[z] = [v / b for v in Pd_z[z]]
    t = sum(Pz)
    Pz = [v / t for v in Pz]
    return Pz, Pd_z, Pw_z


def joint_probability(Pz, Pd_z, Pw_z):
    r"""Eq. (2): :math:`P(d,w) = \sum_z P(z)P(d\mid z)P(w\mid z)`."""
    K = len(Pz)
    D, V = len(Pd_z[0]), len(Pw_z[0])
    return [[sum(Pz[z] * Pd_z[z][d] * Pw_z[z][w] for z in range(K))
             for w in range(V)] for d in range(D)]


def log_likelihood(n_dw, Pz, Pd_z, Pw_z):
    r""":math:`\sum_{d,w} n(d,w)\log P(d,w)`.

    EM cannot decrease it, which is what the anchor checks.
    """
    N, D, V = _check(n_dw)
    P = joint_probability(Pz, Pd_z, Pw_z)
    tot = 0.0
    for d in range(D):
        for w in range(V):
            if N[d][w] > 0.0:
                tot += N[d][w] * math.log(max(P[d][w], _EPS))
    return tot


def fit_plsa(n_dw, K, iters=100, tol=1e-8, seed=0):
    r"""EM to a fixed point, recording the likelihood each sweep."""
    N, D, V = _check(n_dw)
    if int(K) < 1:
        raise ValueError("plsa: K must be at least 1")
    rng = np.random.default_rng(seed)

    def norm(v):
        s = sum(v)
        return [x / s for x in v]

    Pz = norm([0.5 + float(rng.uniform()) for _ in range(int(K))])
    Pd_z = [norm([0.5 + float(rng.uniform()) for _ in range(D)])
            for _ in range(int(K))]
    Pw_z = [norm([0.5 + float(rng.uniform()) for _ in range(V)])
            for _ in range(int(K))]
    hist, prev = [], None
    it = 0
    for it in range(1, int(iters) + 1):
        post = e_step(N, Pz, Pd_z, Pw_z)
        Pz, Pd_z, Pw_z = m_step(N, post, int(K))
        ll = log_likelihood(N, Pz, Pd_z, Pw_z)
        hist.append(ll)
        if prev is not None and abs(ll - prev) < float(tol):
            break
        prev = ll
    return RichResult(payload={
        "estimate": Pw_z, "P_z": Pz, "P_d_given_z": Pd_z,
        "P_w_given_z": Pw_z, "loglik_history": hist,
        "final_loglik": hist[-1], "iterations": it,
        "K": int(K), "n_docs": D, "vocab": V,
        "n_parameters": int(K) * (D + V) + int(K),
        "method": "EM for the aspect model; Hofmann (1999) eqs. "
                  "(3)-(6)",
        "caveat": "P(z|d) is a per-document PARAMETER, so the count "
                  "grows with the corpus and an unseen document needs "
                  "EM re-run -- the gap LDA's Dirichlet prior closes",
    })


def perplexity(n_dw, Pz, Pd_z, Pw_z):
    r""":math:`\exp(-\text{loglik}/\sum n)`."""
    N, _, _ = _check(n_dw)
    tot = sum(sum(r) for r in N)
    return math.exp(-log_likelihood(N, Pz, Pd_z, Pw_z) / tot)


def cheatsheet():
    return ("plsa: the ASPECT model. P(d,w) = sum_z P(z)P(d|z)P(w|z) "
            "-- d and w independent GIVEN z, with |z| small so z is a "
            "bottleneck. EM: E step is Bayes over z, M step is "
            "expected-count normalisation. Fixes LSA's missing "
            "likelihood and gives non-negative parameters. Does NOT "
            "fix generalisation: P(z|d) is a per-document parameter, "
            "so parameters grow with the corpus -- that is what LDA's "
            "Dirichlet prior removes.")


# compact alias per ledger/NAMING.md
probabilisticlsa = fit_plsa

# public names resolved by fn/_lazy_map.json
plsa = fit_plsa
