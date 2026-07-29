# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The UMAP objective: fuzzy-neighbourhood cross-entropy, minimised
by gradient descent on a small embedding (McInnes et al. 2018;
Alammar Ch 5)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_umap_projection"]


def _lcg_stream(seed, n):
    s = int(seed) % 2 ** 32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out[i] = (s + 0.5) / 2 ** 32
    return out


def alammar_umap_projection(X, n_neighbors=5, min_dist=0.1, d_out=2,
                            n_steps=200, learning_rate=0.05, seed=1):
    """Minimise CE(fuzzy graph of X, fuzzy graph of Z) over Z.

    High-dimensional weights: w_ij = exp(-(d_ij - rho_i)/sigma_i) over
    the k-NN, symmetrised by probabilistic union; low-dimensional
    kernel 1/(1 + a d^2) with a from min_dist. Full-batch gradient
    descent from an LCG start. The payload reports the objective
    BEFORE and AFTER, and the tests require it to fall -- a projector
    that does not reduce its own objective is a random number
    generator with axes.

    References: Alammar and Grootendorst, Ch 5; McInnes, Healy and
    Melville (2018). Simplified: exact k-NN, fixed sigma solved by
    bisection, full-batch descent -- stated, not hidden.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    k = int(n_neighbors)
    if not 2 <= k < n:
        raise ValueError(f"n_neighbors must lie in [2, {n - 1}].")
    dd = int(d_out)
    if dd < 1:
        raise ValueError("d_out must be positive.")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    order = np.argsort(D, axis=1)
    W = np.zeros((n, n))
    log2k = np.log2(k)
    for i in range(n):
        nb = order[i, 1:k + 1]
        dists = D[i, nb]
        rho = dists[0]
        lo, hi = 1e-8, 1e4
        for _ in range(64):
            sig = 0.5 * (lo + hi)
            s = np.exp(-(np.maximum(dists - rho, 0.0)) / sig).sum()
            if s > log2k:
                hi = sig
            else:
                lo = sig
        sig = 0.5 * (lo + hi)
        W[i, nb] = np.exp(-(np.maximum(dists - rho, 0.0)) / sig)
    P = W + W.T - W * W.T          # probabilistic union
    a = 1.0 / (float(min_dist) ** 2 + 1e-12) if min_dist > 0 else 100.0

    u = _lcg_stream(seed, n * dd)
    Z = (u.reshape(n, dd) - 0.5) * 10.0

    def objective(Z):
        dz2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
        Q = 1.0 / (1.0 + a * dz2)
        np.fill_diagonal(Q, 1.0)
        eps = 1e-12
        ce = -(P * np.log(Q + eps) + (1 - P) * np.log(1 - Q + eps))
        np.fill_diagonal(ce, 0.0)
        return float(ce.sum())

    obj0 = objective(Z)
    lr = float(learning_rate)
    for _ in range(int(n_steps)):
        diff = Z[:, None, :] - Z[None, :, :]
        dz2 = (diff ** 2).sum(axis=2)
        Q = 1.0 / (1.0 + a * dz2)
        eps = 1e-12
        # dCE/d(dz2): attractive from P, repulsive from (1-P)
        coeff = (P * a * Q - (1 - P) * a * Q * Q / (1 - Q + eps))
        np.fill_diagonal(coeff, 0.0)
        grad = 2.0 * (coeff[:, :, None] * diff).sum(axis=1)
        Z = Z - lr * grad
    obj1 = objective(Z)
    return RichResult(payload={
        "embedding": [[float(v) for v in r] for r in Z],
        "objective_initial": obj0, "objective_final": obj1,
        "objective_decreased": obj1 < obj0,
        "estimate": obj1, "n": n,
        "method": "UMAP fuzzy cross-entropy, full-batch descent "
                  "(McInnes et al. 2018, simplified)"})


def cheatsheet():
    return "alumap: fuzzy k-NN graph CE minimised by descent, objective reported"
