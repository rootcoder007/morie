# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-negative matrix factorisation (ESL Ch 14.6)."""

from math import log as math_log

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["esl_nmf"]


def _kl(X, WH):
    """Generalised Kullback-Leibler divergence sum x log(x / wh) - x + wh (0 log 0 = 0)."""
    Xl, Wl = X.tolist(), WH.tolist()
    d = 0.0
    for xr, wr in zip(Xl, Wl):
        for x, w in zip(xr, wr):
            d += (x * math_log(x / w) if x > 0 else 0.0) - x + w
    return d


def _lcg(count, seed):
    s = int(seed)
    out = np.empty(count)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out[i] = (s + 0.5) / 2 ** 32
    return out


def esl_nmf(X, k, max_iter=500, tol=1e-10, seed=13, loss="frobenius", W0=None, H0=None):
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

    loss : {"frobenius", "kl"}
        ``"kl"`` maximises the Poisson log-likelihood of ESL eq 14.73,
        :math:`\\sum_{ij}[x_{ij}\\log(WH)_{ij} - (WH)_{ij}]`, by the updates of eq
        14.74 (equivalently minimises the generalised KL divergence).
    W0, H0 : array-like, optional
        Starting factors (n x k, k x p); by default a deterministic LCG start.

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

    >>> from morie.fn import _array_core as np
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
    if loss not in ("frobenius", "kl"):
        raise ValueError("loss must be 'frobenius' or 'kl'")
    scale = float(np.sqrt(X.mean() / k)) if X.mean() > 0 else 1.0
    W = (_lcg(n * k, seed).reshape(n, k) + 0.1) * scale if W0 is None else np.asarray(W0, dtype=float).reshape(n, k)
    H = (_lcg(k * p, seed + 1).reshape(k, p) + 0.1) * scale if H0 is None else np.asarray(H0, dtype=float).reshape(k, p)
    eps = 1e-12
    normX = float(np.linalg.norm(X)) or 1.0
    prev = float("inf")
    converged, it = False, 0
    path = []
    for it in range(1, int(max_iter) + 1):
        if loss == "frobenius":
            H = H * (W.T @ X) / (W.T @ W @ H + eps)
            W = W * (X @ H.T) / (W @ H @ H.T + eps)
            err = float(np.linalg.norm(X - W @ H))
            done = abs(prev - err) <= tol * normX
        else:
            # ESL eq 14.74 (Lee & Seung 2001): multiplicative updates that never
            # increase the Kullback-Leibler divergence, W first, then H
            W = W * ((X / (W @ H + eps)) @ H.T) / (np.ones((n, p)) @ H.T + eps)
            H = H * (W.T @ (X / (W @ H + eps))) / (W.T @ np.ones((n, p)) + eps)
            err = _kl(X, W @ H)
            done = abs(prev - err) <= tol * (1.0 + err)
        path.append(err)
        if done:
            converged = True
            break
        prev = err
    err = float(np.linalg.norm(X - W @ H))
    WH = W @ H
    extra = {}
    if loss == "kl":
        extra = {"kl_divergence": _kl(X, WH), "loglik": float(np.sum(X * np.log(WH + 1e-300) - WH)), "divergence_path": path}
    return RichResult(payload={**extra,
        "estimate": err / normX, "W": [float(v) for v in W.ravel()],
        "H": [float(v) for v in H.ravel()], "frobenius_error": err,
        "relative_error": err / normX, "iterations": int(it),
        "converged": bool(converged), "n": int(n), "p": int(p), "k": k,
        "loss": loss,
        "method": "NMF by Lee-Seung multiplicative updates; non-unique, start-dependent"})


def cheatsheet():
    return "eslnmf: parts-based because nothing cancels; non-convex and scale-nonunique"


# compact alias per ledger/NAMING.md
eslnmf = esl_nmf
