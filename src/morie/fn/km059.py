# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.6: the Kronecker product used by KronA."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_kronecker_product"]


def _kron(A, B):
    """Validated Kronecker product; km061/km062 import this."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("both factors must be 2-D matrices.")
    if A.size == 0 or B.size == 0:
        raise ValueError("a Kronecker factor is empty.")
    return np.kron(A, B), A, B


def kamath_ch4_kronecker_product(A, B):
    """W = A (x) B: the (m b1, n b2) block matrix whose (i, j) block is
    a_ij B, for A of shape (m, n).

    Unlike a rank decomposition this imposes NO low-rank assumption --
    rank(A (x) B) = rank(A) rank(B), which is why KronA can be
    full-rank at LoRA's parameter count. The rank is reported.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.6, printed
    p. 153.

    Examples
    --------
    >>> out = kamath_ch4_kronecker_product([[1.0, 2.0]],
    ...                                    [[1.0, 0.0], [0.0, 1.0]])
    >>> out["W"]
    [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]
    >>> out["shape"]
    (2, 4)
    """
    W, Am, Bm = _kron(A, B)
    return RichResult(payload={
        "W": [[float(v) for v in row] for row in W],
        "shape": (int(W.shape[0]), int(W.shape[1])),
        "rank": int(np.linalg.matrix_rank(W)),
        "n_params": int(Am.size + Bm.size),
        "estimate": float(W[0, 0]), "n": int(W.size),
        "method": "Kronecker product (Kamath Eq 4.6)"})


def cheatsheet():
    return "km059: W = A (x) B, block matrix a_ij B, rank multiplies"
