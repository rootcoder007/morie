# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-negative matrix factorization."""

from . import _array_core as np

from ._richresult import RichResult
from .nmf import nmf as _nmf

__all__ = ["geron_nmf"]


def geron_nmf(X, n_components=2, max_iter=400, tol=1e-6, seed=42):
    """
    Non-negative matrix factorization.

    Formula: X approx W H, W >= 0, H >= 0

    The multiplicative-update fit is DELEGATED to the finished
    implementation :func:`morie.fn.nmf.nmf`; this wrapper adds the
    relative error and the non-negativity audit. The constraint is the
    whole point: with no cancellation allowed, the parts have to ADD up
    to the whole, so the factors come out as additive parts (Lee and
    Seung's faces made of noses and eyebrows) instead of the signed,
    uninterpretable directions PCA returns.

    The objective is non-convex, so the fit depends on ``seed``; a
    different seed is a different local minimum, not a bug.

    Parameters
    ----------
    X : array-like, shape (m, p)
        Non-negative matrix.
    n_components : int, default 2
        Inner rank k.
    max_iter : int, default 400
    tol : float, default 1e-6
    seed : int, default 42

    Returns
    -------
    result : RichResult
        Keys: W, H, reconstruction, reconstruction_error,
        relative_error, n_iter, estimate, n, method.

    Examples
    --------
    A rank-1 non-negative matrix is recovered essentially exactly:

    >>> r = geron_nmf([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], 1)
    >>> bool(r["relative_error"] < 1e-6)
    True
    >>> bool((r["W"] >= 0).all() and (r["H"] >= 0).all())
    True
    >>> r["reconstruction"].shape
    (3, 2)

    A negative entry is rejected outright:

    >>> geron_nmf([[1.0, -1.0]], 1)
    Traceback (most recent call last):
        ...
    ValueError: geron_nmf: X must be non-negative; 1 entry is below zero

    References
    ----------
    Geron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_nmf: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_nmf: X contains non-finite values")
    neg = int(np.sum(A < 0))
    if neg:
        raise ValueError(f"geron_nmf: X must be non-negative; {neg} entr{'y is' if neg == 1 else 'ies are'} below zero")
    k = int(n_components)
    if not (1 <= k <= min(A.shape)):
        raise ValueError(f"geron_nmf: n_components must lie in [1, {min(A.shape)}], got {n_components!r}")

    res = _nmf(A, n_components=k, max_iter=int(max_iter), tol=float(tol), seed=int(seed))
    W = np.asarray(res.extra["W"], dtype=float)
    H = np.asarray(res.extra["H"], dtype=float)
    recon = W @ H
    err = float(np.linalg.norm(A - recon, "fro"))
    denom = float(np.linalg.norm(A, "fro"))
    rel = err / denom if denom > 0 else err
    return RichResult(
        title="Non-negative matrix factorization",
        summary_lines=[("Components", k), ("Frobenius error", err), ("Relative error", rel)],
        interpretation="Non-negativity forbids cancellation, so the factors read as additive parts.",
        payload={
            "W": W,
            "H": H,
            "reconstruction": recon,
            "reconstruction_error": err,
            "relative_error": rel,
            "n_iter": int(res.extra["n_iter"]),
            "n_components": k,
            "estimate": err,
            "n": int(A.shape[0]),
            "method": "NMF by multiplicative updates, delegated to morie.fn.nmf.nmf",
        },
    )


def cheatsheet():
    return "hmnmf: Non-negative matrix factorization X ~ W H"


# compact alias per ledger/NAMING.md
geronnmf = geron_nmf
