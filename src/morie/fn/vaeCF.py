# morie.fn -- function file (rootcoder007/morie)
"""Multinomial variational autoencoder for collaborative filtering.

SOURCE.  Liang, D., Krishnan, R.G., Hoffman, M.D. and Jebara, T. (2018),
"Variational Autoencoders for Collaborative Filtering", *Proceedings of
the 2018 World Wide Web Conference* (WWW '18), pp. 689-698,
doi:10.1145/3178876.3186150.

Their contribution is the MULTINOMIAL likelihood in place of the
Gaussian or logistic ones (their Section 2.2):

    pi(z_u) = softmax( f_theta(z_u) ),
    log p(x_u | z_u) = sum_i x_ui log pi_i(z_u),

which spends its fixed probability budget on the items the model
believes will be clicked -- the fit to the ranking loss is the whole
argument of the paper.  The objective is the beta-annealed bound of
their Eq. (5),

    L_beta(x_u) = E_q[ log p(x_u | z) ] - beta * KL( q(z|x_u) || p(z) ),

with beta < 1 the "partial regularisation" they recommend in
Section 2.2.2 (they report beta around 0.2 as best on their data).

The encoder input is the L2-normalised log(1 + x_u) transform of their
Section 2.2 / Section 4, not the raw counts.

RANKING.  Recall@K and truncated NDCG@K are their Section 4.1
definitions,

    Recall@K(u, w) = sum_{r<=K} I[w(r) in I_u] / min(K, |I_u|)
    DCG@K(u, w)    = sum_{r<=K} (2^{I[w(r) in I_u]} - 1) / log2(r + 1)

with NDCG@K the DCG divided by its ideal value.  Ranking ties are broken
by ascending item index in both language arms -- R scans a matrix
column-major and Python row-major, so an unpinned tie rule is a real
parity hazard rather than a hypothetical one.

DETERMINISM.  Untrained weights and the reparameterisation noise come
from the shared deterministic normal stream, so both arms hold the same
numbers.  Relevance defaults to the input clicks, which makes the
reported metrics in-sample; pass ``relevance`` for a held-out set.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vae_cf"]


def vae_cf(R, K=5, latent_dim=2, beta=0.2, n_samples=16, relevance=None,
           w_scale=1.0, skip=0):
    """Mult-VAE ELBO and top-K ranking metrics.

    Parameters
    ----------
    R : array-like
        n_users-by-n_items click matrix, non-negative.
    K : int
        Ranking cut-off, 1 <= K <= n_items.
    latent_dim : int
        Latent width, >= 1.
    beta : float
        KL annealing weight, >= 0.
    n_samples : int
        Reparameterised draws per user, >= 1.
    relevance : array-like or None
        Binary held-out relevance, same shape as ``R``.  ``None`` uses
        ``R > 0``.
    w_scale : float
        Scales the decoder weights.  ``0`` makes the logits constant, so
        pi is uniform -- the closed-form anchor.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    RichResult
        ``elbo``, ``loglik``, ``kl``, ``elbo_per_user``,
        ``loglik_per_user``, ``kl_per_user``, ``recall``, ``ndcg``,
        ``recall_per_user``, ``ndcg_per_user``, ``ranking`` (1-based
        item indices), ``mu``, ``logvar``, ``n_users``, ``n_items``,
        ``k``.

    Raises
    ------
    ValueError
        Empty or ragged ``R``, negative entries, K outside 1..n_items,
        non-positive ``latent_dim`` or ``n_samples``, a negative
        ``beta``, or a ``relevance`` of the wrong shape.

    References
    ----------
    Liang, D., Krishnan, R.G., Hoffman, M.D. and Jebara, T. (2018).
    WWW '18, pp. 689-698.  doi:10.1145/3178876.3186150.
    """
    X = core.mat(R)
    nu = len(X)
    if nu == 0:
        raise ValueError("vae_cf: R is empty")
    ni = len(X[0])
    for r in X:
        if len(r) != ni:
            raise ValueError("vae_cf: rows of R have unequal length")
        for v in r:
            if v < 0.0:
                raise ValueError("vae_cf: R must be non-negative")
    K = int(K)
    if K < 1 or K > ni:
        raise ValueError("vae_cf: K must lie in 1 .. n_items")
    m = int(latent_dim)
    if m < 1:
        raise ValueError("vae_cf: latent_dim must be positive")
    L = int(n_samples)
    if L < 1:
        raise ValueError("vae_cf: n_samples must be positive")
    beta = float(beta)
    if beta < 0.0:
        raise ValueError("vae_cf: beta must be non-negative")
    skip = int(skip)
    if skip < 0:
        raise ValueError("vae_cf: skip must be non-negative")
    if relevance is None:
        R0 = [[1.0 if X[u][i] > 0.0 else 0.0 for i in range(ni)] for u in range(nu)]
    else:
        R0 = core.mat(relevance)
        if len(R0) != nu or any(len(r) != ni for r in R0):
            raise ValueError("vae_cf: relevance must have the same shape as R")
        R0 = [[1.0 if v > 0.0 else 0.0 for v in r] for r in R0]
    # Section 2.2 input transform: L2-normalised log(1 + x)
    Xn = []
    for u in range(nu):
        row = [math.log1p(X[u][i]) for i in range(ni)]
        nrm = 0.0
        for v in row:
            nrm += v * v
        nrm = math.sqrt(nrm)
        Xn.append([v / nrm for v in row] if nrm > 0.0 else [0.0] * ni)
    Wm = vc.draw(ni, m, skip, 1.0 / math.sqrt(ni))
    Wl = vc.draw(ni, m, skip + ni * m, 0.1 / math.sqrt(ni))
    Wd = vc.draw(m, ni, skip + 2 * ni * m, float(w_scale) / math.sqrt(m))
    eps = vc.draw(L, m, skip + 2 * ni * m + m * ni, 1.0)
    mu = core.matmul(Xn, Wm)
    lv = core.matmul(Xn, Wl)
    llu = [0.0] * nu
    klu = [0.0] * nu
    rank = []
    rec = [0.0] * nu
    ndc = [0.0] * nu
    for u in range(nu):
        sig = [math.exp(0.5 * lv[u][j]) for j in range(m)]
        t = 0.0
        for j in range(m):
            t += mu[u][j] * mu[u][j] + sig[j] * sig[j] - 1.0 - lv[u][j]
        klu[u] = 0.5 * t
        acc = 0.0
        for l in range(L):
            z = [mu[u][j] + sig[j] * eps[l][j] for j in range(m)]
            lg = [0.0] * ni
            for i in range(ni):
                s = 0.0
                for j in range(m):
                    s += z[j] * Wd[j][i]
                lg[i] = s
            mx = lg[0]
            for v in lg:
                if v > mx:
                    mx = v
            se = 0.0
            for v in lg:
                se += math.exp(v - mx)
            lse = mx + math.log(se)
            s = 0.0
            for i in range(ni):
                if X[u][i] > 0.0:
                    s += X[u][i] * (lg[i] - lse)
            acc += s
        llu[u] = acc / L
        lgm = [0.0] * ni
        for i in range(ni):
            s = 0.0
            for j in range(m):
                s += mu[u][j] * Wd[j][i]
            lgm[i] = s
        # tie rule pinned: descending score, then ascending item index
        idx = sorted(range(ni), key=lambda i: (-lgm[i], i))
        rank.append([i + 1 for i in idx])
        nrel = 0
        for i in range(ni):
            nrel += int(R0[u][i] > 0.0)
        hit = 0
        dcg = 0.0
        for r in range(K):
            h = 1.0 if R0[u][idx[r]] > 0.0 else 0.0
            hit += int(h)
            dcg += (2.0 ** h - 1.0) / math.log(r + 2.0, 2.0)
        den = K if K < nrel else nrel
        rec[u] = hit / den if den > 0 else 0.0
        ide = 0.0
        for r in range(den):
            ide += 1.0 / math.log(r + 2.0, 2.0)
        ndc[u] = dcg / ide if ide > 0.0 else 0.0
    per = [llu[u] - beta * klu[u] for u in range(nu)]
    ll = sum(llu) / nu
    kl = sum(klu) / nu
    return RichResult(
        title="Mult-VAE for collaborative filtering",
        summary_lines=[("users", nu), ("items", ni), ("NDCG@K", sum(ndc) / nu)],
        payload={
            "estimate": ll - beta * kl,
            "elbo": ll - beta * kl,
            "loglik": ll,
            "kl": kl,
            "elbo_per_user": per,
            "loglik_per_user": llu,
            "kl_per_user": klu,
            "recall": sum(rec) / nu,
            "ndcg": sum(ndc) / nu,
            "recall_per_user": rec,
            "ndcg_per_user": ndc,
            "ranking": rank,
            "mu": mu,
            "logvar": lv,
            "n_users": nu,
            "n_items": ni,
            "k": K,
            "method": "Mult-VAE: multinomial likelihood with beta-annealed KL (Liang et al. 2018 Secs. 2.2, 2.2.2, 4.1)",
        },
    )


def cheatsheet():
    return "vaeCF: multinomial VAE for collaborative filtering (Liang et al. 2018)"

# public names resolved by fn/_lazy_map.json
vaecf = vae_cf
