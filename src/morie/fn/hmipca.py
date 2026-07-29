# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Incremental PCA for out-of-core / streaming datasets."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_incremental_pca"]

_METHOD = "Incremental PCA (streaming co-moment accumulation)"


def geron_incremental_pca(X_iter, n_components, batch_size=None):
    """
    Incremental PCA for out-of-core / streaming datasets.

    Formula: update PCs with mini-batches

    The point of incremental PCA is that the full ``m x n`` matrix never
    has to be in memory at once.  What *is* kept is the running mean and
    the running co-moment matrix ``S = sum (x - mu)(x - mu)^T``, both
    updated with Chan's parallel formula when each batch arrives:

    ``S <- S_a + S_b + (n_a n_b / n) * delta delta^T``

    where ``delta`` is the difference of the two means.  The naive
    alternative -- accumulating ``sum x x^T`` and subtracting
    ``n mu mu^T`` at the end -- loses catastrophic precision when the
    data are far from the origin, which is precisely the situation
    out-of-core data tend to be in.

    Because the co-moment is exact, the components equal those of a
    single-pass PCA on the whole dataset to floating-point precision,
    whatever the batch size.  Memory is ``O(n^2)`` rather than
    ``O(m n)``.

    Parameters
    ----------
    X_iter : iterable of array-like, or array-like
        Batches of rows.  An ordinary 2-D array is split into batches of
        ``batch_size`` rows.
    n_components : int
        Components to keep, ``1 <= n_components <= n_features``.
    batch_size : int, optional
        Rows per batch when ``X_iter`` is a single array.  Defaults to
        the whole array.

    Returns
    -------
    result : RichResult
        Keys: components, explained_variance, explained_variance_ratio,
        mean, n_samples_seen, n_batches, estimate, n, method.

    Examples
    --------
    Batching does not change the answer.  The same data in one batch and
    in three give the same explained variance:

    >>> X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    >>> one = geron_incremental_pca(X, n_components=1)
    >>> three = geron_incremental_pca(X, n_components=1, batch_size=2)
    >>> bool(np.allclose(one["explained_variance"], three["explained_variance"]))
    True
    >>> three["n_batches"], three["n_samples_seen"]
    (3, 6)

    The mean is exact and the first component is the x-axis:

    >>> [float(v) for v in three["mean"]]
    [3.5, 0.0]
    >>> [round(abs(float(v)), 9) for v in three["components"][0]]
    [1.0, 0.0]

    Sample variance of 1..6 is 3.5, and that is the explained variance:

    >>> round(float(three["explained_variance"][0]), 9)
    3.5

    All the variance is on one axis:

    >>> round(float(three["explained_variance_ratio"][0]), 9)
    1.0

    References
    ----------
    Géron Ch 7
    """
    # Normalise the input into a list of 2-D batches.
    batches = []
    arr = None
    if isinstance(X_iter, np.ndarray):
        arr = X_iter
    elif not hasattr(X_iter, "__iter__"):
        raise ValueError("geron_incremental_pca: X_iter must be an iterable of batches or a 2-D array")
    else:
        try:
            candidate = np.asarray(X_iter, dtype=float)
        except (ValueError, TypeError):
            candidate = None
        if candidate is not None and candidate.ndim == 2:
            arr = candidate
        else:
            batches = [np.atleast_2d(np.asarray(b, dtype=float)) for b in X_iter]

    if arr is not None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"geron_incremental_pca: X_iter must be a non-empty 2-D array, got shape {arr.shape}")
        bs = arr.shape[0] if batch_size is None else int(batch_size)
        if bs < 1:
            raise ValueError(f"geron_incremental_pca: batch_size must be at least 1, got {batch_size!r}")
        batches = [arr[i : i + bs] for i in range(0, arr.shape[0], bs)]

    if not batches:
        raise ValueError("geron_incremental_pca: no batches were supplied")

    n_feat = batches[0].shape[1]
    mean = np.zeros(n_feat)
    S = np.zeros((n_feat, n_feat))
    seen = 0
    for b_idx, B in enumerate(batches):
        if B.ndim != 2 or B.size == 0:
            raise ValueError(f"geron_incremental_pca: batch {b_idx} is not a non-empty 2-D array (shape {B.shape})")
        if B.shape[1] != n_feat:
            raise ValueError(
                f"geron_incremental_pca: batch {b_idx} has {B.shape[1]} features but the first batch had {n_feat}"
            )
        if not np.all(np.isfinite(B)):
            raise ValueError(f"geron_incremental_pca: batch {b_idx} contains non-finite values")
        nb = B.shape[0]
        mb = B.mean(axis=0)
        Cb = B - mb
        Sb = Cb.T @ Cb
        if seen == 0:
            mean, S, seen = mb, Sb, nb
        else:
            delta = mb - mean
            total = seen + nb
            S = S + Sb + np.outer(delta, delta) * (seen * nb / total)
            mean = mean + delta * (nb / total)
            seen = total

    d = int(n_components)
    if not (1 <= d <= n_feat):
        raise ValueError(f"geron_incremental_pca: n_components must lie in 1..{n_feat}, got {n_components!r}")
    if seen < 2:
        raise ValueError(
            f"geron_incremental_pca: needs at least 2 samples for a sample covariance, saw {seen}"
        )

    cov = S / (seen - 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    total_var = float(np.sum(vals))
    if total_var == 0:
        raise ValueError("geron_incremental_pca: the data have zero variance in every direction")

    components = vecs[:, :d].T
    explained = vals[:d]
    ratio = explained / total_var

    return RichResult(
        title="Incremental PCA",
        summary_lines=[
            ("Samples seen", int(seen)),
            ("Batches", len(batches)),
            ("Components", d),
            ("Variance explained", float(np.sum(ratio))),
        ],
        interpretation=(
            "The co-moment update is exact, so batching changes nothing but the memory profile: "
            "O(n^2) instead of O(m n)."
        ),
        payload={
            "components": components,
            "explained_variance": explained,
            "explained_variance_ratio": ratio,
            "mean": mean,
            "covariance": cov,
            "n_samples_seen": int(seen),
            "n_batches": len(batches),
            "estimate": float(np.sum(ratio)),
            "n": int(seen),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmipca: incremental PCA via exact streaming co-moment updates (batch size cannot change the answer)"
