# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-negative matrix factorization reconstruction objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_nmf_objective"]

_METHOD = "NMF Frobenius reconstruction objective"


def geron_nmf_objective(X, W, H):
    r"""Squared Frobenius reconstruction error of a non-negative factorization.

    .. math::
        \min_{W \ge 0,\; H \ge 0} \|X - WH\|_F^2

    The constraint is the point.  PCA's components subtract as freely as
    they add, so its "parts" cancel each other and are unreadable; forbid
    negatives and the only way to build :math:`X` is by *adding* pieces,
    which is why NMF factors of face images look like noses and eyebrows.
    Non-negativity is enforced here rather than assumed -- a negative
    entry silently makes the result a plain low-rank fit and not NMF at
    all.  The objective is non-convex in ``(W, H)`` jointly (convex in
    each alone), so this scores a factorization; it does not find one.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Non-negative data matrix.
    W : array-like, shape (m, k)
    H : array-like, shape (k, n)

    Returns
    -------
    RichResult
        Payload keys ``objective``, ``reconstruction``, ``residual``,
        ``relative_error``, ``rank``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, NMF section.

    Examples
    --------
    An exact rank-1 factorization has objective 0:

    >>> X = [[1.0, 2.0], [2.0, 4.0]]
    >>> r = geron_nmf_objective(X, [[1.0], [2.0]], [[1.0, 2.0]])
    >>> r["objective"]
    0.0

    Halving H leaves half the matrix behind:

    >>> h = geron_nmf_objective(X, [[1.0], [2.0]], [[0.5, 1.0]])
    >>> round(h["objective"], 6)
    6.25

    Negative factors are not NMF:

    >>> geron_nmf_objective(X, [[-1.0], [2.0]], [[1.0, 2.0]])
    Traceback (most recent call last):
        ...
    ValueError: W has negative entries; NMF requires W >= 0.
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    Wm = np.atleast_2d(np.asarray(W, dtype=float))
    Hm = np.atleast_2d(np.asarray(H, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    for name, M in (("X", A), ("W", Wm), ("H", Hm)):
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contains non-finite values.")
    if np.any(A < 0):
        raise ValueError("X has negative entries; NMF requires X >= 0.")
    if np.any(Wm < 0):
        raise ValueError("W has negative entries; NMF requires W >= 0.")
    if np.any(Hm < 0):
        raise ValueError("H has negative entries; NMF requires H >= 0.")
    if Wm.shape[0] != A.shape[0]:
        raise ValueError(f"W has {Wm.shape[0]} rows but X has {A.shape[0]}.")
    if Hm.shape[1] != A.shape[1]:
        raise ValueError(f"H has {Hm.shape[1]} columns but X has {A.shape[1]}.")
    if Wm.shape[1] != Hm.shape[0]:
        raise ValueError(
            f"W has inner dimension {Wm.shape[1]} but H has {Hm.shape[0]}."
        )

    R = Wm @ Hm
    E = A - R
    obj = float(np.sum(E**2))
    denom = float(np.sum(A**2))

    return RichResult(
        title="NMF objective",
        summary_lines=[("||X - WH||_F^2", obj), ("Rank", int(Wm.shape[1]))],
        payload={
            "objective": obj,
            "reconstruction": R.tolist(),
            "residual": E.tolist(),
            "relative_error": float(np.sqrt(obj / denom)) if denom > 0 else 0.0,
            "rank": int(Wm.shape[1]),
            "estimate": obj,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grnmfo: ||X - WH||_F^2 with W,H >= 0 enforced; additive parts, non-convex jointly"
