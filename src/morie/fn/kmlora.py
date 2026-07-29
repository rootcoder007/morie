# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LoRA: low-rank adaptation of a frozen weight matrix."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_lora_weight_update"]


def kamath_lora_weight_update(W0, A, B, alpha, r, x):
    """h = W_0 x + (alpha / r) B A x, with W_0 in R^{d x k},
    A in R^{r x k}, B in R^{d x r}.

    The rank is taken from the shapes and CHECKED against ``r`` rather
    than trusted: (alpha / r) with the wrong r rescales the whole
    adapter, and nothing downstream would notice.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, LoRA (Hu et al.
    2021).

    Examples
    --------
    >>> out = kamath_lora_weight_update([[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0, 0.0]], [[0.0], [2.0]], alpha=4.0, r=1, x=[3.0, 5.0])
    >>> out["h"]
    [3.0, 29.0]
    >>> out["n_trainable"], out["n_frozen"]
    (4, 4)
    """
    W0 = np.atleast_2d(np.asarray(W0, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    r = int(r)
    alpha = float(alpha)
    if r < 1:
        raise ValueError(f"the rank r must be at least 1; got {r}.")
    d, k = W0.shape
    if A.shape != (r, k):
        raise ValueError(
            f"A must be (r, k) = ({r}, {k}); got {A.shape}.")
    if B.shape != (d, r):
        raise ValueError(
            f"B must be (d, r) = ({d}, {r}); got {B.shape}.")
    if x.size != k:
        raise ValueError(
            f"x must have {k} entries to multiply a {d}x{k} weight; "
            f"got {x.size}.")
    scale = alpha / r
    base = W0 @ x
    delta = scale * (B @ (A @ x))
    h = base + delta
    return RichResult(payload={
        "h": [float(v) for v in h],
        "base": [float(v) for v in base],
        "delta": [float(v) for v in delta],
        "estimate": float(h[0]), "scaling": scale,
        "rank": r, "alpha": alpha,
        "n_trainable": int(A.size + B.size),
        "n_frozen": int(W0.size),
        "n": int(h.size),
        "method": "LoRA forward h = W0 x + (alpha/r) B A x"})


def cheatsheet():
    return "kmlora: W0 x + (alpha/r) B A x with rank checked against shapes"
