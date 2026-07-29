# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.11: MoverScore n-gram distance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_moverscore_distance"]


def kamath_ch8_moverscore_distance(x_i, y_j, E=None):
    r"""d(x_i^n, y_j^n) = || E(x_i^n) - E(y_j^n) ||_2.

    With ``E`` given it must be a callable embedding function applied
    to each argument; without it, ``x_i`` and ``y_j`` are taken to be
    the embeddings already. This is also the whole of Eq 8.14
    (Sentence Mover's Distance), so ``morie.fn.km126`` delegates here.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.11, printed
    p. 326.

    Examples
    --------
    >>> out = kamath_ch8_moverscore_distance([0.0, 0.0], [3.0, 4.0])
    >>> out["estimate"]
    5.0
    """
    if E is not None:
        if not callable(E):
            raise ValueError("E must be a callable embedding function "
                             "or None.")
        ex = np.atleast_1d(np.asarray(E(x_i), dtype=float)).ravel()
        ey = np.atleast_1d(np.asarray(E(y_j), dtype=float)).ravel()
    else:
        ex = np.atleast_1d(np.asarray(x_i, dtype=float)).ravel()
        ey = np.atleast_1d(np.asarray(y_j, dtype=float)).ravel()
    if ex.size == 0 or ey.size == 0:
        raise ValueError("an empty embedding has no distance.")
    if ex.size != ey.size:
        raise ValueError(
            f"embedding widths differ: {ex.size} vs {ey.size}.")
    if not (np.all(np.isfinite(ex)) and np.all(np.isfinite(ey))):
        raise ValueError("the embeddings contain non-finite values.")
    diff = ex - ey
    return RichResult(payload={
        "estimate": float(np.linalg.norm(diff)),
        "difference": [float(v) for v in diff], "n": int(ex.size),
        "method": "MoverScore Euclidean n-gram distance "
                  "(Kamath Eq 8.11)"})


def cheatsheet():
    return "km123: L2 distance between two n-gram embeddings"
