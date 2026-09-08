# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.11: LoftQ's quantisation-plus-low-rank objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_loftq_objective"]


def kamath_ch4_loftq_objective(W, Q, A, B):
    """min_{Q,A,B} ||W - Q - A B^T||_F.

    The Frobenius norm of what quantisation plus the low-rank factors
    FAIL to reproduce. This scores a candidate (Q, A, B); the
    minimisation over them is the caller's search. Smaller is better,
    and 0 means the triple reproduces W exactly.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.11, printed
    p. 160.

    Examples
    --------
    >>> out = kamath_ch4_loftq_objective([[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0, 0.0], [0.0, 0.0]], [[0.0], [1.0]], [[0.0], [1.0]])
    >>> out["estimate"]
    0.0
    >>> kamath_ch4_loftq_objective([[3.0, 0.0], [0.0, 4.0]],
    ...     [[0.0, 0.0], [0.0, 0.0]], [[0.0], [0.0]],
    ...     [[0.0], [0.0]])["estimate"]
    5.0
    """
    Wm = np.atleast_2d(np.asarray(W, dtype=float))
    Qm = np.atleast_2d(np.asarray(Q, dtype=float))
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = np.atleast_2d(np.asarray(B, dtype=float))
    if Qm.shape != Wm.shape:
        raise ValueError(f"Q is {Qm.shape} but W is {Wm.shape}.")
    if Am.shape[1] != Bm.shape[1]:
        raise ValueError(
            f"A is {Am.shape} and B is {Bm.shape}; A B^T needs a shared "
            "rank dimension.")
    low = Am @ Bm.T
    if low.shape != Wm.shape:
        raise ValueError(
            f"A B^T is {low.shape} but W is {Wm.shape}.")
    resid = Wm - Qm - low
    return RichResult(payload={
        "estimate": float(np.linalg.norm(resid, "fro")),
        "residual": [[float(v) for v in row] for row in resid],
        "quantisation_error": float(np.linalg.norm(Wm - Qm, "fro")),
        "r": int(Am.shape[1]), "n": int(Wm.size),
        "method": "LoftQ Frobenius objective (Kamath Eq 4.11)"})


def cheatsheet():
    return "km064: ||W - Q - A B^T||_F, smaller is better"
