# morie.fn -- function file (rootcoder007/morie)
r"""Latent Dirichlet Allocation by variational EM.

A document is a mixture over topics; a topic is a distribution over
words. The generative story is short:

1. choose :math:`\theta \sim \mathrm{Dir}(\alpha)`;
2. for each of the :math:`N` words, choose a topic
   :math:`z_n \sim \mathrm{Multinomial}(\theta)`, then a word
   :math:`w_n \sim p(w_n \mid z_n, \beta)`.

with :math:`\beta` a :math:`k \times V` matrix,
:math:`\beta_{ij} = p(w^j = 1 \mid z^i = 1)`. The document length
:math:`N` is ancillary -- it is drawn from a Poisson in the paper's
statement but nothing downstream depends on that, so its randomness is
ignored.

**Why the posterior is intractable.** Marginalising :math:`\theta` and
:math:`z` couples them through :math:`\beta`, so
:math:`p(\theta, z \mid w, \alpha, \beta)` has no closed form.

**The variational fix is a deliberate act of vandalism.** Delete the
edges between :math:`\theta`, :math:`z` and :math:`w`, drop the
:math:`w` nodes, and give the wreckage its own free parameters. The
resulting family factorises, and the best member of it is the one
closest in KL divergence to the true posterior (eq. 5). Setting the
derivatives of that divergence to zero gives a fixed point:

.. math:: \phi_{ni} &\propto \beta_{i w_n}
            \exp\{E_q[\log \theta_i \mid \gamma]\}, \\
          \gamma_i &= \alpha_i + \sum_{n=1}^{N} \phi_{ni},

(eqs. 6-7) with the expectation available in closed form,

.. math:: E_q[\log\theta_i \mid \gamma]
          = \Psi(\gamma_i) - \Psi\Big(\sum_{j=1}^{k}\gamma_j\Big),

(eq. 8) where :math:`\Psi` is the digamma function.

**Both updates are recognisable.** The :math:`\gamma` update is a
posterior Dirichlet given the expected topic counts
:math:`E[z_n \mid \phi_n]` -- prior plus observations. The
:math:`\phi` update is Bayes' theorem,
:math:`p(z_n\mid w_n) \propto p(w_n \mid z_n)p(z_n)`, with the prior
replaced by the exponential of the expected log topic weight. Not the
expected weight -- the *exponential of the expected log*, which is
smaller by Jensen, and that gap is exactly what makes variational
inference under-confident about rare topics.

**The variational parameters are per-document.** The optimisation is
run for fixed :math:`w`, so :math:`(\gamma^*, \phi^*)` are functions
of the document. :math:`\gamma^*(w)` is what represents that document
in the topic simplex.

**Why the bound only ever goes up.** Each update maximises the same
lower bound over one block with the other fixed, so the bound is
monotone across iterations. That is a property of the algorithm and
the anchor checks it directly rather than checking that the fit looks
plausible.

References
----------
Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) "Latent Dirichlet
Allocation", *Journal of Machine Learning Research* 3, 993-1022.
Sec. 3 (the generative process, the k x V matrix beta, and the
ancillary role of N), Sec. 5.1 (the variational family obtained by
dropping edges, eq. (5) as a KL minimisation, the fixed-point updates
of eqs. (6)-(7), the digamma expression of eq. (8), and the
interpretation of both updates), and the variational inference
algorithm initialising phi at 1/k and gamma at alpha + N/k.

Hofmann, T. (1999) "Probabilistic Latent Semantic Analysis",
*Proceedings of the Fifteenth Conference on Uncertainty in Artificial
Intelligence (UAI 1999)*, 289-296, arXiv:1301.6705. The aspect model
LDA places a Dirichlet prior over; implemented in :mod:`plsa`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["e_log_theta", "variational_inference", "elbo",
           "variational_em", "topic_words"]

_EPS = 1e-12


def e_log_theta(gamma):
    r"""Eq. (8): :math:`\Psi(\gamma_i) - \Psi(\sum_j \gamma_j)`."""
    g = [float(v) for v in k.vec(gamma)]
    if any(v <= 0.0 for v in g):
        raise ValueError("lda: gamma must be strictly positive, got "
                         "%r" % (min(g),))
    s = k.digamma(sum(g))
    return [k.digamma(v) - s for v in g]


def variational_inference(doc, alpha, beta, iters=100, tol=1e-8):
    r"""Eqs. (6)-(7) iterated to a fixed point for one document.

    ``doc`` is a list of word indices. Initialised as the paper's
    algorithm does: :math:`\phi_{ni} = 1/k` and
    :math:`\gamma_i = \alpha_i + N/k`.
    """
    w = [int(v) for v in doc]
    B = [[float(v) for v in r] for r in k.mat(beta)]
    K = len(B)
    if K < 1:
        raise ValueError("lda: beta must have at least one topic")
    V = len(B[0])
    if any(v < 0 or v >= V for v in w):
        raise ValueError("lda: a word index is outside the "
                         "vocabulary of %d" % V)
    N = len(w)
    if N < 1:
        raise ValueError("lda: the document is empty")
    a = ([float(alpha)] * K if isinstance(alpha, (int, float))
         else [float(v) for v in k.vec(alpha)])
    if len(a) != K:
        raise ValueError("lda: alpha has %d entries for %d topics"
                         % (len(a), K))
    if any(v <= 0.0 for v in a):
        raise ValueError("lda: alpha must be strictly positive")
    phi = [[1.0 / K] * K for _ in range(N)]
    gam = [a[i] + N / float(K) for i in range(K)]
    it, conv = 0, False
    for it in range(1, int(iters) + 1):
        elog = e_log_theta(gam)
        new = []
        for n in range(N):
            row = [B[i][w[n]] * math.exp(elog[i]) for i in range(K)]
            z = sum(row)
            if z <= _EPS:
                raise ValueError("lda: word %d has zero probability "
                                 "under every topic" % w[n])
            new.append([v / z for v in row])
        ng = [a[i] + sum(new[n][i] for n in range(N))
              for i in range(K)]
        delta = max(abs(ng[i] - gam[i]) for i in range(K))
        phi, gam = new, ng
        if delta < float(tol):
            conv = True
            break
    return {"phi": phi, "gamma": gam, "iterations": it,
            "converged": conv, "K": K, "N": N,
            "topic_proportions": [v / sum(gam) for v in gam]}


def elbo(doc, alpha, beta, phi, gamma):
    r"""The variational lower bound on :math:`\log p(w \mid \alpha,
    \beta)`.

    Each update maximises this over one block, so it can only rise --
    which is what makes a decrease a bug rather than noise.
    """
    w = [int(v) for v in doc]
    B = [[float(v) for v in r] for r in k.mat(beta)]
    K, N = len(B), len(w)
    a = ([float(alpha)] * K if isinstance(alpha, (int, float))
         else [float(v) for v in k.vec(alpha)])
    g = [float(v) for v in k.vec(gamma)]
    elog = e_log_theta(g)
    val = k.lgamma(sum(a)) - sum(k.lgamma(v) for v in a)
    val += sum((a[i] - 1.0) * elog[i] for i in range(K))
    for n in range(N):
        for i in range(K):
            p = phi[n][i]
            if p <= _EPS:
                continue
            val += p * elog[i]
            val += p * math.log(max(B[i][w[n]], _EPS))
            val -= p * math.log(p)
    val -= k.lgamma(sum(g)) - sum(k.lgamma(v) for v in g)
    val -= sum((g[i] - 1.0) * elog[i] for i in range(K))
    return val


def variational_em(docs, K, V, alpha=0.1, iters=30, inner=50,
                   seed=0, tol=1e-6):
    r"""Alternate variational inference with the :math:`\beta` update.

    The M step is the expected word-topic count, normalised:
    :math:`\beta_{ij} \propto \sum_d \sum_n \phi^d_{ni}
    w^{dj}_n`.
    """
    D = [[int(v) for v in d] for d in docs]
    if not D:
        raise ValueError("lda: no documents given")
    if int(K) < 1 or int(V) < 1:
        raise ValueError("lda: K and V must be at least 1")
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(int(K)):
        row = [0.1 + float(rng.uniform()) for _ in range(int(V))]
        z = sum(row)
        B.append([v / z for v in row])
    hist, prev = [], None
    for it in range(1, int(iters) + 1):
        counts = [[_EPS] * int(V) for _ in range(int(K))]
        total = 0.0
        for d in D:
            if not d:
                continue
            r = variational_inference(d, alpha, B, iters=inner)
            total += elbo(d, alpha, B, r["phi"], r["gamma"])
            for n, wn in enumerate(d):
                for i in range(int(K)):
                    counts[i][wn] += r["phi"][n][i]
        for i in range(int(K)):
            z = sum(counts[i])
            B[i] = [v / z for v in counts[i]]
        hist.append(total)
        if prev is not None and abs(total - prev) < float(tol):
            break
        prev = total
    return RichResult(payload={
        "estimate": B, "beta": B, "elbo_history": hist,
        "final_elbo": hist[-1] if hist else float("nan"),
        "K": int(K), "V": int(V), "n_docs": len(D),
        "iterations": len(hist),
        "method": "variational EM; Blei, Ng & Jordan (2003) "
                  "Sec. 5.1, eqs. (6)-(8)",
    })


def topic_words(beta, n_top=5, vocab=None):
    r"""The most probable words in each topic."""
    B = [[float(v) for v in r] for r in k.mat(beta)]
    out = []
    for i in range(len(B)):
        idx = sorted(range(len(B[i])), key=lambda j: -B[i][j])
        idx = idx[:int(n_top)]
        out.append([(vocab[j] if vocab else j, B[i][j])
                    for j in idx])
    return out


def cheatsheet():
    return ("lda: theta ~ Dir(alpha), z_n ~ Mult(theta), w_n ~ "
            "p(.|z_n, beta). The posterior is intractable because "
            "theta and z couple through beta, so DELETE those edges "
            "and fit the wreckage by minimising KL (eq. 5). Fixed "
            "point: phi_ni prop beta_{i,w_n} exp(E_q[log theta_i]), "
            "gamma_i = alpha_i + sum_n phi_ni, with E_q[log theta_i] "
            "= Psi(gamma_i) - Psi(sum gamma). Note it is exp(E[log]) "
            "not E[.] -- smaller by Jensen, which is why variational "
            "inference under-weights rare topics. The bound only "
            "rises.")


# compact alias per ledger/NAMING.md
latentdirichlet = variational_em

# public names resolved by fn/_lazy_map.json
lda_topic = variational_em
ldatopic = variational_em
