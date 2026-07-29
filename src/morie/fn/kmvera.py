# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VeRA: shared frozen random matrices with per-layer learned scaling
vectors."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_vera_adapter"]


def kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x):
    """h = W_0 x + Lambda_b B Lambda_d A x, with A and B frozen random
    matrices and Lambda_b, Lambda_d learned diagonals.

    Only the two DIAGONALS are trained -- d + r numbers per layer
    instead of LoRA's r(d + k) -- which is the entire claim of the
    method, so the trainable count is reported next to LoRA's for the
    same shapes. A and B are shared across layers by construction;
    nothing here writes to them.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, VeRA
    (Kopiczko et al. 2024).

    Examples
    --------
    >>> out = kamath_vera_adapter([[0.0, 0.0], [0.0, 0.0]],
    ...     [[1.0, 0.0]], [[1.0], [1.0]], [3.0, 4.0], [2.0], [1.0, 0.0])
    >>> out["h"]
    [6.0, 8.0]
    >>> out["n_trainable"], out["n_trainable_lora_equivalent"]
    (3, 4)
    """
    W0 = np.atleast_2d(np.asarray(W0, dtype=float))
    A = np.atleast_2d(np.asarray(A_frozen, dtype=float))
    B = np.atleast_2d(np.asarray(B_frozen, dtype=float))
    lb = np.atleast_1d(np.asarray(lam_b, dtype=float)).ravel()
    ld = np.atleast_1d(np.asarray(lam_d, dtype=float)).ravel()
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    d, k = W0.shape
    r = A.shape[0]
    if A.shape[1] != k:
        raise ValueError(
            f"A must be (r, k) with k = {k}; got {A.shape}.")
    if B.shape != (d, r):
        raise ValueError(
            f"B must be (d, r) = ({d}, {r}); got {B.shape}.")
    if ld.size != r:
        raise ValueError(
            f"Lambda_d is the r-dimensional diagonal: expected {r} "
            f"entries, got {ld.size}.")
    if lb.size != d:
        raise ValueError(
            f"Lambda_b is the d-dimensional diagonal: expected {d} "
            f"entries, got {lb.size}.")
    if x.size != k:
        raise ValueError(f"x must have {k} entries; got {x.size}.")
    base = W0 @ x
    delta = lb * (B @ (ld * (A @ x)))
    h = base + delta
    return RichResult(payload={
        "h": [float(v) for v in h],
        "base": [float(v) for v in base],
        "delta": [float(v) for v in delta],
        "estimate": float(h[0]),
        "rank": int(r),
        "n_trainable": int(lb.size + ld.size),
        "n_trainable_lora_equivalent": int(A.size + B.size),
        "n_frozen": int(W0.size + A.size + B.size),
        "n": int(h.size),
        "method": "VeRA h = W0 x + Lambda_b B Lambda_d A x"})


def cheatsheet():
    return "kmvera: only the two diagonals train; A, B frozen and shared"
