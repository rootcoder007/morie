# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian random projection matrix scaled by 1/sqrt(d')."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gaussian_rand_projection"]

_METHOD = "Gaussian random projection"


def geron_gaussian_rand_projection(X, d_out, seed=0):
    """
    Gaussian random projection matrix scaled by 1/sqrt(d').

    Formula: X' = X * R, R_ij ~ N(0, 1/d')

    The scaling ``1/sqrt(d_out)`` is what makes the projection an
    isometry *in expectation*: with ``R_ij ~ N(0, 1/d_out)``,
    ``E||R^T u||^2 = ||u||^2`` for every fixed ``u``, so squared
    distances are preserved on average and the distortion is what the
    Johnson-Lindenstrauss bound controls.  The realised distortion on
    the supplied data is measured and returned rather than assumed.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Data; a 1-D array is treated as a single row.
    d_out : int
        Target dimensionality, ``1 <= d_out``.
    seed : int
        Seed for the projection matrix.

    Returns
    -------
    result : RichResult
        Keys: X_projected, R, d_in, d_out, max_distortion,
        mean_distortion, estimate, n, method.

    Examples
    --------
    The projection is exactly ``X @ R`` and the entries of ``R`` carry
    the ``1/sqrt(d_out)`` scale, so an identity-like check is exact:

    >>> r = geron_gaussian_rand_projection([[1.0, 0.0], [0.0, 1.0]], d_out=3, seed=0)
    >>> r["X_projected"].shape, r["R"].shape
    ((2, 3), (2, 3))
    >>> import numpy as np
    >>> bool(np.allclose(r["X_projected"], r["R"]))
    True

    Squared distances survive the projection: with 200 input dimensions
    and 100 output dimensions the worst pairwise distortion stays small:

    >>> rng = np.random.default_rng(3)
    >>> Z = rng.normal(size=(20, 200))
    >>> big = geron_gaussian_rand_projection(Z, d_out=100, seed=1)
    >>> big["d_in"], big["d_out"]
    (200, 100)
    >>> wide = geron_gaussian_rand_projection(Z, d_out=400, seed=1)
    >>> narrow = geron_gaussian_rand_projection(Z, d_out=5, seed=1)
    >>> bool(wide["max_distortion"] < big["max_distortion"] < narrow["max_distortion"])
    True

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2:
        raise ValueError(f"geron_gaussian_rand_projection: X must be 1-D or 2-D, got ndim={A.ndim}")
    if A.size == 0:
        raise ValueError("geron_gaussian_rand_projection: X is empty")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_gaussian_rand_projection: X contains non-finite values")
    m, d_in = A.shape
    k = int(d_out)
    if k < 1:
        raise ValueError(f"geron_gaussian_rand_projection: d_out must be at least 1, got {k}")

    rng = np.random.default_rng(int(seed))
    R = rng.normal(loc=0.0, scale=1.0 / np.sqrt(k), size=(d_in, k))
    Z = A @ R

    # Realised distortion of pairwise squared distances (only meaningful
    # with at least two rows).
    if m >= 2:
        iu, ju = np.triu_indices(m, k=1)
        d_before = np.sum((A[iu] - A[ju]) ** 2, axis=1)
        d_after = np.sum((Z[iu] - Z[ju]) ** 2, axis=1)
        keep = d_before > 0
        if np.any(keep):
            ratio = d_after[keep] / d_before[keep]
            max_dist = float(np.max(np.abs(ratio - 1.0)))
            mean_dist = float(np.mean(np.abs(ratio - 1.0)))
        else:
            max_dist = mean_dist = 0.0
    else:
        max_dist = mean_dist = float("nan")

    return RichResult(
        title="Gaussian random projection",
        summary_lines=[
            ("Input dimension", d_in),
            ("Output dimension", k),
            ("Worst squared-distance distortion", max_dist),
        ],
        interpretation=(
            "Distortion shrinks as d_out grows; geron_johnson_lindenstrauss gives the d_out "
            "needed for a target epsilon."
        ),
        payload={
            "X_projected": Z,
            "R": R,
            "d_in": int(d_in),
            "d_out": k,
            "max_distortion": max_dist,
            "mean_distortion": mean_dist,
            "estimate": mean_dist,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmgrp: Gaussian random projection X' = X R with R_ij ~ N(0, 1/d'), plus realised distortion"
