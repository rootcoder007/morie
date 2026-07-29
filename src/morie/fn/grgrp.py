# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian random projection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gaussian_random_projection"]

_METHOD = "Gaussian random projection, R_ij ~ N(0, 1/d)"


def _lcg_normals(count, seed):
    """``count`` standard normals from the reference LCG via Box-Muller.

    The LCG is ``s = (1664525 s + 1013904223) mod 2**32``,
    ``u = (s + 0.5) / 2**32`` -- the same stream the tranche's tests
    use, so a reported variance can be checked by hand.
    """
    s = int(seed) % 2**32
    n_pairs = (count + 1) // 2
    out = np.empty(2 * n_pairs, dtype=float)
    for i in range(n_pairs):
        s = (1664525 * s + 1013904223) % 2**32
        u1 = (s + 0.5) / 2**32
        s = (1664525 * s + 1013904223) % 2**32
        u2 = (s + 0.5) / 2**32
        rad = np.sqrt(-2.0 * np.log(u1))
        out[2 * i] = rad * np.cos(2.0 * np.pi * u2)
        out[2 * i + 1] = rad * np.sin(2.0 * np.pi * u2)
    return out[:count]


def geron_gaussian_random_projection(X, d, seed=0):
    r"""Project onto a random Gaussian subspace.

    .. math::
        Z = X R,\qquad R_{ij} \sim \mathcal N(0, 1/d)

    The variance ``1/d`` is the whole trick: summing ``d`` such terms
    gives a dot product whose expectation is the original one, so
    squared distances are preserved *in expectation* without any
    rescaling afterwards.  ``mean_distance_ratio`` reports how close
    that came on the data actually supplied.

    Draws come from the deterministic LCG above rather than
    ``numpy.random``, so the same ``seed`` gives the same projection in
    every process, and ``achieved_variance`` -- the sample variance of
    the entries of ``R`` -- can be compared against the target ``1/d``.

    Parameters
    ----------
    X : array-like, shape (m, n)
    d : int
        Target dimension, at least 1.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``projected``, ``R``, ``target_variance`` (``1/d``),
        ``achieved_variance``, ``mean_distance_ratio``,
        ``max_distance_ratio``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, Gaussian RP section.  See :mod:`morie.fn.grjll` for how
    large ``d`` has to be for a distortion guarantee.

    Examples
    --------
    Shapes and the variance target:

    >>> X = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    >>> r = geron_gaussian_random_projection(X, d=2, seed=7)
    >>> len(r["projected"]), len(r["projected"][0])
    (3, 2)
    >>> r["target_variance"]
    0.5

    The projection is a pure matrix product, so row 0 of the result is
    row 0 of ``R`` (X being the identity here):

    >>> [round(v, 10) for v in r["projected"][0]] == [round(v, 10) for v in r["R"][0]]
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, n), got shape {A.shape}.")
    if A.size == 0:
        raise ValueError("X is empty.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    d = int(d)
    if d < 1:
        raise ValueError(f"d must be a positive integer, got {d}.")
    m, n = A.shape

    R = _lcg_normals(n * d, seed).reshape(n, d) * np.sqrt(1.0 / d)
    Z = A @ R

    if m > 1:
        i, j = np.triu_indices(m, k=1)
        d0 = np.linalg.norm(A[i] - A[j], axis=1)
        d1 = np.linalg.norm(Z[i] - Z[j], axis=1)
        keep = d0 > 0
        ratio = d1[keep] / d0[keep] if np.any(keep) else np.array([])
    else:
        ratio = np.array([])

    return RichResult(
        title="Gaussian random projection",
        summary_lines=[("n -> d", f"{n} -> {d}"),
                       ("Var target", 1.0 / d),
                       ("Var achieved", float(np.var(R)))],
        payload={
            "projected": Z.tolist(),
            "R": R.tolist(),
            "target_variance": 1.0 / d,
            "achieved_variance": float(np.var(R, ddof=1)) if R.size > 1 else 0.0,
            "mean_distance_ratio": float(ratio.mean()) if ratio.size else None,
            "max_distance_ratio": float(ratio.max()) if ratio.size else None,
            "d": d,
            "seed": int(seed),
            "estimate": Z.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgrp: Z = X R with R_ij ~ N(0, 1/d) from the reference LCG; reports achieved variance"
