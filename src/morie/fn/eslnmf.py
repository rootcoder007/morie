# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-negative matrix factorisation (ESL Ch 14.6)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_nmf"]


def _lcg(count, seed):
    s = int(seed)
    out = np.empty(count)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out[i] = (s + 0.5) / 2 ** 32
    return out


def esl_nmf(X, k, max_iter=500, tol=1e-10, seed=13):
    """
    NMF: X ~ W H with W, H >= 0, by Lee-Seung multiplicative updates.

    ESL Ch 14.6 contrasts NMF with PCA: because nothing may be
    negative, the factors can only ADD, which forces a
    parts-based rather than cancelling representation. That is the
    appeal and also the catch — the objective is not jointly convex,
    so the answer depends on the start, and NMF is NOT unique
    (W D and D^-1 H fit equally well for any positive diagonal D).
    Both facts are reported rather than left for the user to discover.

    The multiplicative updates preserve non-negativity automatically
    from a non-negative start, which is why they are used instead of
    projected gradient. Initialisation is from the shared LCG so runs
    reproduce exactly.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Non-negative data.
    k : int
        Rank, 1 <= k <= min(n, p).
    max_iter, tol
        Update controls; tol is on relative Frobenius error change.
    seed : int
        LCG seed for the initialisation.

    Returns
    -------
    result : dict
        Keys: estimate (relative Frobenius error), W (row-major n x k),
        H (row-major k x p), frobenius_error, relative_error,
        iterations, converged, n, p, k, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 14.6 (Eq. 14.74);
    Lee & Seung (2001).

    Examples
    --------
    An exactly rank-1 non-negative matrix is recovered essentially
    exactly, and every entry of both factors stays non-negative:

    >>> import numpy as np
    >>> X = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
    >>> out = esl_nmf(X, 1)
    >>> out["relative_error"] < 1e-6
    True
    >>> min(out["W"]) >= 0.0 and min(out["H"]) >= 0.0
    True

    The reconstruction matches even though W and H individually are
    only determined up to a positive scaling:

    >>> W = np.asarray(out["W"]).reshape(3, 1)
    >>> H = np.asarray(out["H"]).reshape(1, 2)
    >>> bool(np.allclose(W @ H, X, atol=1e-5))
    True
    >>> esl_nmf([[1.0, -1.0]], 1)
    Traceback (most recent call last):
        ...
    ValueError: NMF needs a non-negative matrix; found a negative entry.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    k = int(k)
    if np.any(X < 0):
        raise ValueError("NMF needs a non-negative matrix; found a negative entry.")
    kmax = min(n, p)
    if not 1 <= k <= kmax:
        raise ValueError(f"k must lie in [1, {kmax}]; got {k}.")
    scale = float(np.sqrt(X.mean() / k)) if X.mean() > 0 else 1.0
    W = (_lcg(n * k, seed).reshape(n, k) + 0.1) * scale
    H = (_lcg(k * p, seed + 1).reshape(k, p) + 0.1) * scale
    eps = 1e-12
    normX = float(np.linalg.norm(X)) or 1.0
    prev = float("inf")
    converged, it = False, 0
    for it in range(1, int(max_iter) + 1):
        H = H * (W.T @ X) / (W.T @ W @ H + eps)
        W = W * (X @ H.T) / (W @ H @ H.T + eps)
        err = float(np.linalg.norm(X - W @ H))
        if abs(prev - err) <= tol * normX:
            converged = True
            break
        prev = err
    err = float(np.linalg.norm(X - W @ H))
    return RichResult(payload={
        "estimate": err / normX, "W": [float(v) for v in W.ravel()],
        "H": [float(v) for v in H.ravel()], "frobenius_error": err,
        "relative_error": err / normX, "iterations": int(it),
        "converged": bool(converged), "n": int(n), "p": int(p), "k": k,
        "method": "NMF by Lee-Seung multiplicative updates; non-unique, start-dependent"})


def cheatsheet():
    return "eslnmf: parts-based because nothing cancels; non-convex and scale-nonunique"
