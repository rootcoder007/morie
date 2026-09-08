# morie.fn -- function file (rootcoder007/morie)
r"""Funk SVD: factorise only the ratings that exist.

The name is misleading and the misunderstanding matters. A true SVD
requires a **complete** matrix, and a ratings matrix is almost
entirely missing -- 99% or more. Filling the holes with zeros or with
column means and running an SVD does not give a better estimate of the
missing entries; it fits the imputation.

**So do not complete the matrix: fit only the observed cells.**
Minimise

.. math:: \sum_{(u,i)\in\mathcal{K}} (r_{ui} - \mu - b_u - b_i
          - q_i^\top p_u)^2
          + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)

by stochastic gradient descent over the observed set
:math:`\mathcal{K}`. The regulariser is not optional decoration: with
one latent vector per user and per item, an unregularised fit
reproduces the observed ratings and says nothing about the rest.

**Baselines before factors.** :math:`\mu + b_u + b_i` absorbs the fact
that some users rate high and some items are widely liked. Those
effects are large, and if the factors have to represent them they have
less capacity for the interaction that is actually being modelled --
which is what the factors are for.

**Funk's own procedure trained one factor at a time**, fitting feature
:math:`f` to convergence against the residual left by features
:math:`0\ldots f-1`, rather than all :math:`k` jointly. That is a
greedy, deflation-style fit, and it is the thing that makes "Funk SVD"
a distinct recipe rather than a synonym for regularised MF, so
``fit`` exposes ``incremental`` and the anchor compares the two.

References
----------
Funk, S. (2006) "Netflix Update: Try This at Home",
https://sifter.org/~simon/journal/20061211.html. The original
description of the incremental, regularised gradient-descent
factorisation of the observed entries during the Netflix Prize.
NOTE: this is a blog post, not a peer-reviewed paper -- it is the
origin of the method and is cited as such; the published statements
of the same model are the two references below, and the
implementation follows those.

Koren, Y. (2008) "Factorization Meets the Neighborhood: a
Multifaceted Collaborative Filtering Model", *KDD '08*, 426-434,
doi:10.1145/1401890.1401944. [PDF supplied by Vee.] The baseline
decomposition b_ui = mu + b_u + b_i and the regularised squared-error
objective over the OBSERVED ratings, optimised by stochastic gradient
descent.

Koren, Y., Bell, R. & Volinsky, C. (2009) "Matrix Factorization
Techniques for Recommender Systems", *Computer* 42(8), 30-37,
doi:10.1109/MC.2009.263. The statement that earlier work relying on
imputation to fill in the missing ratings is both expensive and prone
to distortion, and that modelling only the observed entries with
regularisation is preferable.

Salakhutdinov, R. & Mnih, A. (2008) "Probabilistic Matrix
Factorization", *NIPS 2007*, 1257-1264. The probabilistic reading of
the same objective.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["global_mean", "predict", "sgd_epoch", "fit",
           "imputed_svd_error", "rmse"]

_EPS = 1e-12


def global_mean(ratings):
    r""":math:`\mu` over the OBSERVED entries only."""
    R = list(ratings)
    if not R:
        raise ValueError("funkM: no ratings given")
    return sum(float(r) for _, _, r in R) / len(R)


def predict(mu, b_user, b_item, p_u, q_i):
    r""":math:`\hat r_{ui} = \mu + b_u + b_i + q_i^\top p_u`."""
    p = [float(v) for v in k.vec(p_u)]
    q = [float(v) for v in k.vec(q_i)]
    if len(p) != len(q):
        raise ValueError("funkM: the factors differ in width "
                         "(%d, %d)" % (len(p), len(q)))
    return (float(mu) + float(b_user) + float(b_item)
            + sum(p[a] * q[a] for a in range(len(p))))


def sgd_epoch(ratings, mu, bu, bi, P, Q, lr, reg, factor=None):
    r"""One pass over the observed ratings.

    ``factor`` restricts the update to a single latent dimension,
    which is what Funk's incremental schedule does.
    """
    se = 0.0
    for (u, i, r) in ratings:
        u, i = int(u), int(i)
        e = float(r) - predict(mu, bu[u], bi[i], P[u], Q[i])
        se += e * e
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])
        rng = range(len(P[u])) if factor is None else [int(factor)]
        for a in rng:
            pu, qi = P[u][a], Q[i][a]
            P[u][a] = pu + lr * (e * qi - reg * pu)
            Q[i][a] = qi + lr * (e * pu - reg * qi)
    return math.sqrt(se / len(ratings))


def fit(ratings, n_users, n_items, factors=8, epochs=60, lr=0.005,
        reg=0.02, seed=0, incremental=False, epochs_per_factor=20):
    r"""Regularised MF on the observed entries.

    ``incremental=True`` is Funk's own schedule: fit factor 0 to
    convergence, then factor 1 against what it leaves behind, and so
    on.
    """
    R = [(int(u), int(i), float(r)) for u, i, r in ratings]
    if not R:
        raise ValueError("funkM: no ratings given")
    nu, ni, d = int(n_users), int(n_items), int(factors)
    if min(nu, ni, d) < 1:
        raise ValueError("funkM: the counts must be positive")
    if float(reg) < 0.0:
        raise ValueError("funkM: the regularisation cannot be "
                         "negative")
    mu = global_mean(R)
    rng = np.random.default_rng(seed)
    bu = [0.0] * nu
    bi = [0.0] * ni
    P = [[0.1 * (float(rng.uniform()) - 0.5) for _ in range(d)]
         for _ in range(nu)]
    Q = [[0.1 * (float(rng.uniform()) - 0.5) for _ in range(d)]
         for _ in range(ni)]
    hist = []
    if incremental:
        for f in range(d):
            for _ in range(int(epochs_per_factor)):
                hist.append(sgd_epoch(R, mu, bu, bi, P, Q, lr, reg,
                                      factor=f))
    else:
        for _ in range(int(epochs)):
            hist.append(sgd_epoch(R, mu, bu, bi, P, Q, lr, reg))
    return RichResult(payload={
        "estimate": hist[-1], "rmse": hist[-1],
        "rmse_history": hist, "mu": mu, "b_user": bu, "b_item": bi,
        "P": P, "Q": Q, "factors": d,
        "incremental": bool(incremental),
        "observed": len(R),
        "density": len(R) / float(nu * ni),
        "method": "regularised MF on the observed entries; Funk "
                  "(2006), published form in Koren (2008)",
        "note": "the missing entries are never imputed -- only the "
                "observed set is summed over",
    })


def rmse(ratings, mu, bu, bi, P, Q):
    r"""Root mean squared error on a held-out set."""
    R = [(int(u), int(i), float(r)) for u, i, r in ratings]
    if not R:
        raise ValueError("funkM: no ratings to score")
    return math.sqrt(sum((r - predict(mu, bu[u], bi[i], P[u],
                                      Q[i])) ** 2
                         for u, i, r in R) / len(R))


def imputed_svd_error(ratings, n_users, n_items, rank=2,
                      fill="zero"):
    r"""What filling the holes and running an SVD actually gives.

    Kept so the comparison is measurable: the imputation dominates the
    fit, and the reconstruction of the OBSERVED entries is worse than
    a model that never touched the missing ones.
    """
    R = [(int(u), int(i), float(r)) for u, i, r in ratings]
    nu, ni = int(n_users), int(n_items)
    obs = {}
    for u, i, r in R:
        obs[(u, i)] = r
    if fill == "zero":
        base = 0.0
    elif fill == "mean":
        base = global_mean(R)
    else:
        raise ValueError("funkM: fill must be zero or mean, got %r"
                         % (fill,))
    M = [[obs.get((u, i), base) for i in range(ni)]
         for u in range(nu)]
    U, S, Vt = np.linalg.svd(M)
    kk = int(rank)
    approx = [[sum(U[u][t] * S[t] * Vt[t][i]
                   for t in range(min(kk, len(S))))
               for i in range(ni)] for u in range(nu)]
    err = math.sqrt(sum((obs[(u, i)] - approx[u][i]) ** 2
                        for (u, i) in obs) / len(obs))
    return {"rmse_on_observed": err, "fill": fill, "rank": kk,
            "note": "the SVD spent its rank on the imputed cells, "
                    "which outnumber the real ones"}


def cheatsheet():
    return ("funkM: a true SVD needs a COMPLETE matrix and a ratings "
            "matrix is >99% missing -- filling the holes with zeros or "
            "means fits the IMPUTATION, not the data. So sum only over "
            "the OBSERVED entries: minimise (r - mu - b_u - b_i - "
            "q'p)^2 + lambda(||p||^2 + ||q||^2 + b_u^2 + b_i^2) by "
            "SGD. Regularisation is load-bearing -- one vector per user "
            "and item would otherwise just memorise. Baselines "
            "mu + b_u + b_i first, or the factors waste capacity on "
            "effects that are not interactions. Funk's own schedule "
            "trained ONE FACTOR AT A TIME against the previous "
            "residual, which is what makes the recipe distinct.")


# compact alias per ledger/NAMING.md
funk_svd = fit

# public names resolved by fn/_lazy_map.json
funksvd = fit
