# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sparse random projection matrix with {-1,0,+1} entries."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_sparse_rand_projection"]


def geron_sparse_rand_projection(X, d_out, density=None, seed=0):
    """
    Sparse random projection matrix with {-1,0,+1} entries.

    Formula: R_ij = ±sqrt(s/d') with prob 1/(2s), else 0

    Achlioptas / Li sparse projection with ``s = 1/density``. Each entry
    is ``+sqrt(s/d_out)`` or ``-sqrt(s/d_out)`` with probability
    ``1/(2s)`` each and 0 otherwise, which makes ``E[R^T R] = I`` so
    squared distances are preserved in expectation -- the property the
    Johnson-Lindenstrauss lemma quantifies. Li's default density is
    ``1/sqrt(d_in)``. The realised distortion of the pairwise distances is
    measured and returned, not assumed.

    Parameters
    ----------
    X : array-like
        Data (n, d_in).
    d_out : int
        Target dimension (1 <= d_out).
    density : float, optional
        Fraction of non-zeros in (0, 1]; default ``1/sqrt(d_in)``.
    seed : int, default 0
        LCG seed (no global RNG state is touched).

    Returns
    -------
    result : RichResult
        Keys: X_proj, R, density, nnz, max_distortion, mean_distortion,
        estimate, n, method.

    Examples
    --------
    A dense (density = 1) projection has entries exactly ±1/sqrt(d_out),
    so nothing is zero and the shape is as requested:

    >>> import numpy as np
    >>> X = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    >>> r = geron_sparse_rand_projection(X, 2, density=1.0, seed=7)
    >>> r["X_proj"].shape
    (3, 2)
    >>> sorted(set(np.round(np.abs(r["R"]).ravel(), 12).tolist()))
    [0.707106781187]
    >>> int(r["nnz"])
    6

    At density 0.5 roughly half the entries are zero and the non-zeros
    scale up to sqrt(2/d_out) to compensate:

    >>> r2 = geron_sparse_rand_projection(X, 2, density=0.5, seed=3)
    >>> round(float(np.max(np.abs(r2["R"]))), 12)
    1.0
    >>> bool(int(r2["nnz"]) < 6)
    True

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_sparse_rand_projection: X must be a non-empty (n, d_in) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_sparse_rand_projection: X contains non-finite values")
    n, d_in = A.shape
    k = int(d_out)
    if k < 1:
        raise ValueError(f"geron_sparse_rand_projection: d_out must be >= 1, got {k}")
    if k > d_in:
        raise ValueError(
            f"geron_sparse_rand_projection: d_out={k} exceeds the input dimension {d_in}; "
            "a random projection is a reduction"
        )
    dens = 1.0 / np.sqrt(d_in) if density is None else float(density)
    if not (0.0 < dens <= 1.0):
        raise ValueError(f"geron_sparse_rand_projection: density must lie in (0, 1], got {dens}")

    s = 1.0 / dens
    scale = np.sqrt(s / k)
    rng = int(seed) % 2**32
    R = np.zeros((d_in, k))
    for i in range(d_in):
        for j in range(k):
            rng = (1664525 * rng + 1013904223) % 2**32
            u = (rng + 0.5) / 2**32
            if u < 0.5 * dens:
                R[i, j] = scale
            elif u < dens:
                R[i, j] = -scale
    Xp = A @ R

    max_dist = mean_dist = float("nan")
    if n >= 2:
        d0 = []
        d1 = []
        for i in range(n):
            for j in range(i + 1, n):
                a = float(np.linalg.norm(A[i] - A[j]))
                if a > 0:
                    d0.append(a)
                    d1.append(float(np.linalg.norm(Xp[i] - Xp[j])))
        if d0:
            ratio = np.asarray(d1) / np.asarray(d0)
            max_dist = float(np.max(np.abs(ratio - 1.0)))
            mean_dist = float(np.mean(np.abs(ratio - 1.0)))

    return RichResult(
        title="Sparse random projection",
        summary_lines=[
            ("Input dim", int(d_in)),
            ("Output dim", k),
            ("Density", dens),
            ("Non-zeros", int(np.count_nonzero(R))),
            ("Max distance distortion", max_dist),
        ],
        interpretation=(
            "The projection ignores the data entirely -- only the dimensions matter -- so it costs "
            "nothing to fit; sparsity buys speed while keeping E[R^T R] = I."
        ),
        payload={
            "X_proj": Xp,
            "R": R,
            "density": dens,
            "s": s,
            "scale": float(scale),
            "nnz": int(np.count_nonzero(R)),
            "max_distortion": max_dist,
            "mean_distortion": mean_dist,
            "estimate": max_dist,
            "n": int(n),
            "method": "Achlioptas/Li sparse random projection with +/-sqrt(s/d_out) entries",
        },
    )


def cheatsheet():
    return "hmsrp: Sparse random projection matrix with {-1,0,+1} entries"
