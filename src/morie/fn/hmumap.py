# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""UMAP: uniform manifold approximation, preserves local and some global structure."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_umap", "fit_ab"]


def fit_ab(min_dist, spread=1.0):
    """Fit ``(a, b)`` so ``1/(1 + a d^(2b))`` matches UMAP's target curve.

    The target is 1 for ``d <= min_dist`` and ``exp(-(d - min_dist)/spread)``
    beyond it. Fitted by a deterministic grid search on ``log a`` and ``b``
    -- coarse, but reproducible and free of any optimiser dependency.
    """
    d = np.linspace(0.0, 3.0 * spread, 300)
    target = np.where(d <= min_dist, 1.0, np.exp(-(d - min_dist) / spread))
    best = None
    for la in np.linspace(-3.0, 3.0, 121):
        a = float(np.exp(la))
        for b in np.linspace(0.25, 3.0, 56):
            v = 1.0 / (1.0 + a * np.power(np.maximum(d, 1e-12), 2 * b))
            sse = float(np.sum((v - target) ** 2))
            if best is None or sse < best[0]:
                best = (sse, a, float(b))
    return best[1], best[2], best[0]


def geron_umap(X, n_components=2, n_neighbors=3, min_dist=0.1, seed=0, n_iter=300, lr=0.1):
    """
    UMAP: uniform manifold approximation, preserves local and some global structure.

    Formula: fuzzy topological cross-entropy between high-/low-d graphs

    The pipeline is executed in full, exactly (no negative sampling --
    the cross-entropy is summed over all pairs, which is tractable at
    these sizes and removes the stochasticity):

    1. **Local connectivity.** For each point, ``rho_i`` is the distance
       to its nearest neighbour and ``sigma_i`` is solved by binary search
       so that ``sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)``. Subtracting
       rho is what guarantees every point is connected to something -- the
       "uniform manifold" assumption made operational.
    2. **Fuzzy union.** ``W = A + A^T - A * A^T`` symmetrises the directed
       memberships as a probabilistic t-conorm, not an average.
    3. **Low-dimensional membership** ``v = 1/(1 + a*d^(2b))`` with
       ``(a, b)`` fitted from `min_dist` by :func:`fit_ab`.
    4. **Fuzzy cross-entropy** ``sum w log(w/v) + (1-w) log((1-w)/(1-v))``
       minimised by gradient descent. The second term is the repulsion
       that keeps unrelated points apart, and it is why UMAP retains more
       global structure than t-SNE, whose objective has no such term.

    Parameters
    ----------
    X : array-like
        Data (n, d), n >= 3.
    n_components : int, default 2
        Embedding dimension (>= 1).
    n_neighbors : int, default 3
        Neighbourhood size k (2 <= k < n).
    min_dist : float, default 0.1
        Minimum spacing in the embedding (>= 0).
    seed : int, default 0
        LCG seed for the initial embedding.
    n_iter : int, default 300
        Gradient steps (>= 1).
    lr : float, default 0.1
        Learning rate (> 0).

    Returns
    -------
    result : RichResult
        Keys: embedding, graph, a, b, cross_entropy, ce_curve, rho, sigma,
        estimate, n, method.

    Examples
    --------
    Two separated groups: the fuzzy graph has (almost) no weight between
    them, and the cross-entropy falls during optimisation.

    >>> X = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
    >>> r = geron_umap(X, n_components=1, n_neighbors=2, n_iter=200)
    >>> r["embedding"].shape
    (6, 1)
    >>> bool(r["ce_curve"][-1] < r["ce_curve"][0])
    True
    >>> bool(r["graph"][0, 4] < 1e-6)
    True
    >>> bool(r["a"] > 0 and r["b"] > 0)
    True

    Membership of a point in its own nearest neighbour is exactly 1,
    because ``d - rho = 0`` there:

    >>> round(float(r["rho"][0]), 12)
    0.1

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] < 3:
        raise ValueError("geron_umap: X must be 2-D with at least 3 rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_umap: X contains non-finite values")
    n = A.shape[0]
    k = int(n_neighbors)
    if not (2 <= k < n):
        raise ValueError(f"geron_umap: n_neighbors must lie in 2..{n - 1}, got {k}")
    m = int(n_components)
    if m < 1:
        raise ValueError(f"geron_umap: n_components must be >= 1, got {m}")
    md = float(min_dist)
    if not np.isfinite(md) or md < 0:
        raise ValueError(f"geron_umap: min_dist must be non-negative and finite, got {md}")
    it = int(n_iter)
    if it < 1:
        raise ValueError(f"geron_umap: n_iter must be >= 1, got {it}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_umap: lr must be positive and finite, got {eta}")

    diff = A[:, None, :] - A[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    target = np.log2(k)
    P = np.zeros((n, n))
    rho = np.zeros(n)
    sig = np.zeros(n)
    for i in range(n):
        order = np.argsort(D[i], kind="mergesort")[1 : k + 1]
        di = D[i, order]
        rho[i] = float(di[0])
        lo, hi, s = 0.0, np.inf, 1.0
        for _ in range(64):
            w = np.exp(-np.maximum(di - rho[i], 0.0) / s)
            tot = float(np.sum(w))
            if abs(tot - target) < 1e-5:
                break
            if tot > target:
                hi = s
                s = (lo + hi) / 2
            else:
                lo = s
                s = s * 2 if hi == np.inf else (lo + hi) / 2
        sig[i] = s
        P[i, order] = np.exp(-np.maximum(di - rho[i], 0.0) / s)
    W = P + P.T - P * P.T
    np.fill_diagonal(W, 0.0)
    W = np.clip(W, 0.0, 1.0 - 1e-9)

    a, b, ab_sse = fit_ab(md)

    s0 = int(seed) % 2**32
    flat = np.empty(n * m)
    for i in range(n * m):
        s0 = (1664525 * s0 + 1013904223) % 2**32
        flat[i] = (2.0 * ((s0 + 0.5) / 2**32) - 1.0)
    Y = flat.reshape(n, m)

    eps = 1e-9
    ces = []
    off = ~np.eye(n, dtype=bool)
    for _ in range(it + 1):
        dy = Y[:, None, :] - Y[None, :, :]
        s2 = np.sum(dy * dy, axis=2)
        s2 = np.maximum(s2, eps)
        v = 1.0 / (1.0 + a * np.power(s2, b))
        v = np.clip(v, eps, 1.0 - eps)
        ce = float(np.sum(W[off] * np.log(W[off] / v[off] + eps) + (1 - W[off]) * np.log((1 - W[off]) / (1 - v[off]))))
        ces.append(ce)
        if len(ces) > it:
            break
        dce_dv = -W / v + (1.0 - W) / (1.0 - v)
        dv_ds = -a * b * np.power(s2, b - 1.0) / np.square(1.0 + a * np.power(s2, b))
        coef = dce_dv * dv_ds
        np.fill_diagonal(coef, 0.0)
        grad = 2.0 * ((np.diag(coef.sum(axis=1)) - coef) @ Y)
        gn = float(np.max(np.abs(grad)))
        if gn > 4.0:  # clip: the repulsion term is unbounded as v -> 0
            grad = grad * (4.0 / gn)
        Y = Y - eta * grad

    return RichResult(
        title="UMAP embedding",
        summary_lines=[
            ("Points", n),
            ("Neighbours", k),
            ("a, b", f"{a:.4f}, {b:.4f}"),
            ("Final cross-entropy", ces[-1]),
        ],
        interpretation=(
            "UMAP's objective has an explicit repulsive term, so unrelated points are pushed apart "
            "rather than merely not pulled together -- that is what buys it more global structure than t-SNE."
        ),
        payload={
            "embedding": Y,
            "graph": W,
            "directed_graph": P,
            "a": float(a),
            "b": float(b),
            "ab_sse": float(ab_sse),
            "cross_entropy": ces[-1],
            "ce_curve": np.asarray(ces, dtype=float),
            "rho": rho,
            "sigma": sig,
            "estimate": ces[-1],
            "n": int(n),
            "method": "UMAP: smoothed kNN fuzzy graph, fitted (a, b) low-d kernel, exact fuzzy cross-entropy by gradient descent",
        },
    )


def cheatsheet():
    return "hmumap: UMAP: uniform manifold approximation, preserves local and some global structure"
