# morie.fn -- slice s03 (rootcoder007/morie)
"""HDP topic model, the nonparametric LDA.

Source consulted (FETCHED): Teh, Y. W., Jordan, M. I., Beal, M. J. and
Blei, D. M. (2006).  Hierarchical Dirichlet processes.  *JASA* 101(476),
1566-1581.  Section 6.1 applies the HDP to document modelling: each
document is a group, the topics are the shared atoms, and the number of
topics is *not* fixed in advance -- which is the whole difference from
latent Dirichlet allocation (Blei, Ng and Jordan 2003), where K is a
hyperparameter.  The model is the paper's equation (19),

    beta | gamma ~ GEM(gamma),  pi_j | alpha_0, beta ~ DP(alpha_0, beta)
    z_ji | pi_j ~ pi_j,         w_ji | z_ji, phi ~ Mult(phi_(z_ji))

with phi_k a distribution over the vocabulary.

DETERMINISM.  Topic-word distributions are fitted by EM from a symmetric
Dirichlet smoothing prior, with the HDP weights entering the document
side as pseudo-counts.  No Gibbs sampling, so no generator.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .dpsbw import stick_breaking_weights

__all__ = ["hdp_topic_model"]


def hdp_topic_model(docs, gamma=1.0, alpha=1.0, truncation=3, V=None,
                    eta=0.1, max_iter=200, tol=1e-13):
    """Nonparametric topic model over bag-of-words documents.

    Parameters
    ----------
    docs : list of array-like
        Each document as a list of zero-based word ids.
    gamma, alpha : float
        Top-level and document-level concentrations.
    truncation : int
        Number of topics retained.
    V : int, optional
        Vocabulary size; inferred when absent.
    eta : float
        Symmetric Dirichlet smoothing on the topic-word distributions.

    Returns
    -------
    estimate : the log likelihood
    phi      : topic-word distributions, one row per topic
    theta    : document-topic proportions, one row per document
    beta     : the global topic weights
    """
    D = [[int(w) for w in k.vec(d)] for d in docs]
    Vn = int(V) if V is not None else (
        max([max(d) for d in D if d]) + 1 if D else 0)
    K = int(truncation)
    beta = stick_breaking_weights(gamma, K)["pi"]
    tot = 0.0
    for x in beta:
        tot += x
    beta = [x / tot if tot > 0.0 else 1.0 / K for x in beta]
    phi = [[(1.0 + (t * 7 + w * 3) % 5) for w in range(Vn)] for t in range(K)]
    for t in range(K):
        s = 0.0
        for w in range(Vn):
            s += phi[t][w]
        phi[t] = [x / s for x in phi[t]]
    theta = [[beta[t] for t in range(K)] for _ in D]
    ll = float("-inf")
    for _ in range(int(max_iter)):
        cnt = [[0.0] * Vn for _ in range(K)]
        newll = 0.0
        post = []
        for j in range(len(D)):
            acc = [0.0] * K
            for w in D[j]:
                lp = [math.log(theta[j][t] if theta[j][t] > 1e-300 else 1e-300)
                      + math.log(phi[t][w] if phi[t][w] > 1e-300 else 1e-300)
                      for t in range(K)]
                m = k.logsumexp(lp)
                newll += m
                for t in range(K):
                    r = math.exp(lp[t] - m)
                    acc[t] += r
                    cnt[t][w] += r
            post.append(acc)
        for j in range(len(D)):
            nj = float(len(D[j]))
            theta[j] = [(float(alpha) * beta[t] + post[j][t])
                        / (float(alpha) + nj) for t in range(K)]
        for t in range(K):
            s = 0.0
            for w in range(Vn):
                s += cnt[t][w] + float(eta)
            phi[t] = [(cnt[t][w] + float(eta)) / s for w in range(Vn)]
        if abs(newll - ll) < tol:
            ll = newll
            break
        ll = newll
    return RichResult(
        title="HDP topic model",
        summary_lines=[("documents", len(D)), ("topics", K), ("log-lik", ll)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "phi": phi,
            "theta": theta,
            "beta": beta,
            "n_vocab": Vn,
            "method": "HDP topic model, the nonparametric LDA (Teh et al. 2006, sec. 6.1)",
        },
    )


def cheatsheet():
    return "hdptpc: HDP topic model (nonparametric LDA)"


hdptopicmodel = hdp_topic_model
