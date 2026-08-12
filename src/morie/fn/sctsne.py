"""t-SNE embedding (van der Maaten & Hinton 2008)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sctsne", "tsne_embedding"]


def _p_conditional(D2, perplexity, tol=1e-5, max_iter=60):
    # Eq. 1 with per-point binary search on sigma so that the Shannon
    # entropy matches log(perplexity) (the paper's definition of Perp)
    n = len(D2)
    target = math.log(perplexity)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        lo, hi = 1e-20, 1e20
        beta = 1.0                      # 1/(2 sigma^2)
        for _ in range(max_iter):
            num = [math.exp(-D2[i][j] * beta) if j != i else 0.0
                   for j in range(n)]
            s = sum(num)
            if s <= 0:
                s = 1e-300
            pr = [v / s for v in num]
            h = -sum(v * math.log(v) for v in pr if v > 1e-300)
            if abs(h - target) < tol:
                break
            if h > target:
                lo = beta
                beta = beta * 2.0 if hi >= 1e20 else (beta + hi) / 2.0
            else:
                hi = beta
                beta = (beta + lo) / 2.0
        for j in range(n):
            P[i][j] = pr[j]
    return P


def sctsne(X, dim=2, perplexity=10.0, T=300, eta=100.0, seed=0):
    """
    t-distributed stochastic neighbor embedding.

    van der Maaten & Hinton (2008), Algorithm 1 verbatim: compute
    conditional affinities p_{j|i} by Eq. 1 with the per-point
    variance sigma_i binary-searched so the perplexity of P_i equals
    Perp; symmetrize p_ij = (p_{j|i} + p_{i|j}) / (2n); sample the
    initial map from N(0, 1e-4 I); then for T iterations compute the
    Student-t low-dimensional affinities (their Eq. 4)

        q_ij = (1 + |y_i - y_j|^2)^{-1}
               / sum_{k != l} (1 + |y_k - y_l|^2)^{-1},

    the gradient (their Eq. 5)

        dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)
                  (1 + |y_i - y_j|^2)^{-1},

    and update with learning rate eta and momentum alpha(t) = 0.5
    for t < 250, 0.8 after (the paper's schedule).  C is the KL
    divergence KL(P||Q) (Eq. 2).

    Sources
    -------
    van der Maaten, L. & Hinton, G. (2008). Visualizing data using
    t-SNE. *JMLR*, 9, 2579-2605, Eqs. 1-5 and Algorithm 1 (local
    copy fetched-wave3/vandermaaten-hinton-2008-tsne-jmlr9.pdf).

    Parameters
    ----------
    X : matrix (n x d)
        High-dimensional data.
    dim : int
        Output dimensionality.
    perplexity : float
        Perp parameter (roughly the effective neighbor count).
    T : int
        Gradient-descent iterations.
    eta : float
        Learning rate.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: embedding (n x dim), kl (final KL divergence),
        kl_initial, perplexity_error (max deviation of achieved
        entropy from log Perp).
    """
    Xv = [[float(v) for v in row] for row in X]
    n = len(Xv)
    if n < 5:
        raise ValueError("need at least five points")
    if not (1.0 < perplexity < n):
        raise ValueError("perplexity must be in (1, n)")
    D2 = [[sum((a - b) ** 2 for a, b in zip(Xv[i], Xv[j]))
           for j in range(n)] for i in range(n)]
    Pc = _p_conditional(D2, perplexity)
    # perplexity fit check
    perr = 0.0
    for i in range(n):
        h = -sum(v * math.log(v) for v in Pc[i] if v > 1e-300)
        perr = max(perr, abs(h - math.log(perplexity)))
    P = [[(Pc[i][j] + Pc[j][i]) / (2.0 * n) for j in range(n)]
         for i in range(n)]
    rng = np.random.default_rng(seed)
    Y = [[1e-2 * float(rng.normal()) for _ in range(dim)]
         for _ in range(n)]
    Ym1 = [row[:] for row in Y]

    def _q_and_kl():
        W = [[0.0] * n for _ in range(n)]
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = 1.0 / (1.0 + sum((Y[i][k] - Y[j][k]) ** 2
                                     for k in range(dim)))
                W[i][j] = W[j][i] = w
                s += 2.0 * w
        kl = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and P[i][j] > 1e-300:
                    q = max(W[i][j] / s, 1e-300)
                    kl += P[i][j] * math.log(P[i][j] / q)
        return W, s, kl

    _, _, kl0 = _q_and_kl()
    for t in range(1, int(T) + 1):
        W, s, _ = _q_and_kl()
        grads = []
        for i in range(n):
            g = [0.0] * dim
            for j in range(n):
                if i == j:
                    continue
                coef = 4.0 * (P[i][j] - W[i][j] / s) * W[i][j]
                for k in range(dim):
                    g[k] += coef * (Y[i][k] - Y[j][k])
            grads.append(g)
        mom = 0.5 if t < 250 else 0.8
        for i in range(n):
            for k in range(dim):
                new = (Y[i][k] - eta * grads[i][k]
                       + mom * (Y[i][k] - Ym1[i][k]))
                Ym1[i][k] = Y[i][k]
                Y[i][k] = new
    _, _, kl_fin = _q_and_kl()
    return RichResult(payload={
        "embedding": Y,
        "kl": kl_fin,
        "kl_initial": kl0,
        "perplexity_error": perr,
        "T": int(T),
        "seed": int(seed),
        "method": "t-SNE (van der Maaten & Hinton 2008, Alg. 1)",
    })


# long descriptive alias (stub-era name)
tsne_embedding = sctsne


def cheatsheet():
    return "sctsne: P by perplexity search; q ~ Student-t; grad Eq.5 + momentum"
