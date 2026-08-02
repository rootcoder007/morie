# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""t-SNE: KL divergence between joint probabilities in high- and low-dim."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_tsne", "conditional_p"]


def conditional_p(D2, perplexity, tol=1e-5, max_steps=100):
    """Row-wise Gaussian affinities with sigma solved to hit `perplexity`.

    Binary search on beta = 1/(2 sigma^2) until the row entropy equals
    ``log(perplexity)``; this is the step that makes t-SNE adaptive to
    local density instead of using one global bandwidth.
    """
    n = D2.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)
    betas = np.ones(n)
    for i in range(n):
        lo, hi = -np.inf, np.inf
        beta = 1.0
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        Di = D2[i, idx]
        for _ in range(max_steps):
            Pi = np.exp(-Di * beta)
            s = float(np.sum(Pi))
            if s <= 0:
                H = 0.0
                Pi = np.ones_like(Pi) / Pi.size
            else:
                Pi = Pi / s
                H = float(-np.sum(Pi[Pi > 0] * np.log(Pi[Pi > 0])))
            if abs(H - target) < tol:
                break
            if H > target:  # too spread out -> narrow the kernel
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, idx] = Pi
        betas[i] = beta
    return P, betas


def geron_tsne(X, n_components=2, perplexity=5.0, seed=0, n_iter=300, lr=None, momentum=0.8):
    """
    t-SNE: KL divergence between joint probabilities in high- and low-dim.

    Formula: min_Y KL(P || Q); Q uses Student-t heavy tails

    The real algorithm, not a projection dressed up as one:

    1. High-dimensional affinities ``p_j|i`` are Gaussians whose bandwidth
       is solved per point by binary search so every row has the same
       perplexity (:func:`conditional_p`), then symmetrised into
       ``p_ij = (p_j|i + p_i|j) / 2n``.
    2. Low-dimensional affinities use the **Student-t** kernel
       ``q_ij ∝ (1 + ||y_i - y_j||^2)^-1``. The heavy tail is the fix for
       the crowding problem: it lets moderately distant points sit far
       apart without paying an enormous penalty.
    3. ``KL(P || Q)`` is minimised by gradient descent with momentum,
       using the exact gradient
       ``4 * sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||^2)^-1``.

    KL is asymmetric on purpose: putting nearby points far apart costs a
    lot, the reverse costs little -- which is why t-SNE preserves local
    structure and why distances *between* clusters mean nothing.

    Parameters
    ----------
    X : array-like
        Data (n, d), n >= 3.
    n_components : int, default 2
        Embedding dimension (>= 1).
    perplexity : float, default 5.0
        Effective neighbour count; must satisfy ``1 < perplexity < n``.
    seed : int, default 0
        LCG seed for the initial embedding.
    n_iter : int, default 300
        Gradient steps (>= 1).
    lr : float, optional
        Learning rate (> 0). Default ``max(4, n/12)``: the gradient scales
        like 1/n, so a rate tuned for thousands of points diverges on tens.
    momentum : float, default 0.8
        Momentum in [0, 1).

    Returns
    -------
    result : RichResult
        Keys: embedding, P, Q, kl, kl_curve, betas, estimate, n, method.

    Examples
    --------
    Two tight groups far apart stay separated in the embedding, and the
    divergence falls during optimisation:

    >>> import numpy as np
    >>> X = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
    >>> r = geron_tsne(X, n_components=1, perplexity=2.0, n_iter=250)
    >>> r["embedding"].shape
    (6, 1)
    >>> bool(r["kl_curve"][-1] < r["kl_curve"][0])
    True
    >>> within = abs(r["embedding"][0, 0] - r["embedding"][1, 0])
    >>> between = abs(r["embedding"][0, 0] - r["embedding"][4, 0])
    >>> bool(between > within)
    True

    P is a joint distribution: it is symmetric and sums to one.

    >>> round(float(r["P"].sum()), 12)
    1.0
    >>> bool(np.allclose(r["P"], r["P"].T))
    True

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] < 3:
        raise ValueError("geron_tsne: X must be 2-D with at least 3 rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_tsne: X contains non-finite values")
    n = A.shape[0]
    k = int(n_components)
    if k < 1:
        raise ValueError(f"geron_tsne: n_components must be >= 1, got {k}")
    perp = float(perplexity)
    if not (1.0 < perp < n):
        raise ValueError(f"geron_tsne: perplexity must satisfy 1 < perplexity < n ({n}), got {perp}")
    it = int(n_iter)
    if it < 1:
        raise ValueError(f"geron_tsne: n_iter must be >= 1, got {it}")
    eta = max(4.0, n / 12.0) if lr is None else float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_tsne: lr must be positive and finite, got {eta}")
    mom = float(momentum)
    if not (0.0 <= mom < 1.0):
        raise ValueError(f"geron_tsne: momentum must lie in [0, 1), got {mom}")

    diff = A[:, None, :] - A[None, :, :]
    D2 = np.sum(diff * diff, axis=2)
    Pc, betas = conditional_p(D2, perp)
    P = (Pc + Pc.T) / (2.0 * n)
    np.fill_diagonal(P, 0.0)

    s = int(seed) % 2**32
    flat = np.empty(n * k)
    for i in range(n * k):
        s = (1664525 * s + 1013904223) % 2**32
        flat[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * 1e-2
    Y = flat.reshape(n, k)
    V = np.zeros_like(Y)

    kls = []
    for _ in range(it):
        d = Y[:, None, :] - Y[None, :, :]
        num = 1.0 / (1.0 + np.sum(d * d, axis=2))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / np.sum(num), 1e-12)
        np.fill_diagonal(Q, 0.0)
        mask = P > 0
        kl_now = float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))
        if not np.isfinite(kl_now):
            raise ValueError("geron_tsne: the embedding diverged; lower lr")
        kls.append(kl_now)
        W = (P - Q) * num
        grad = 4.0 * ((np.diag(W.sum(axis=1)) - W) @ Y)
        V = mom * V - eta * grad
        Y = Y + V

    d = Y[:, None, :] - Y[None, :, :]
    num = 1.0 / (1.0 + np.sum(d * d, axis=2))
    np.fill_diagonal(num, 0.0)
    Q = np.maximum(num / np.sum(num), 1e-12)
    np.fill_diagonal(Q, 0.0)
    mask = P > 0
    kl = float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))
    kls.append(kl)

    return RichResult(
        title="t-SNE embedding",
        summary_lines=[
            ("Points", n),
            ("Embedding dim", k),
            ("Perplexity", perp),
            ("Final KL(P || Q)", kl),
        ],
        interpretation=(
            "t-SNE preserves neighbourhoods, not distances: cluster sizes and the gaps between clusters "
            "in the picture carry no meaning, and the layout changes with perplexity."
        ),
        payload={
            "embedding": Y,
            "P": P,
            "Q": Q,
            "kl": kl,
            "kl_curve": np.asarray(kls, dtype=float),
            "betas": betas,
            "perplexity": perp,
            "estimate": kl,
            "n": int(n),
            "method": "t-SNE: perplexity-calibrated Gaussian P, Student-t Q, KL minimised by gradient descent with momentum",
        },
    )


def cheatsheet():
    return "hmtsne: t-SNE: KL divergence between joint probabilities in high- and low-dim"
