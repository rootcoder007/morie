# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.5: the LoRA forward pass."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_lora_forward"]


def kamath_ch4_lora_forward(W_0, B, A, x):
    """h = W_0 x + dW x = W_0 x + B A x.

    ``W_0`` is the frozen (d, k) weight, ``B`` (d, r) and ``A`` (r, k)
    the trainable factors, ``x`` a (k,) input. dW = BA has rank at most
    r by construction, and that rank is reported: it is the entire
    assumption LoRA rests on. With B = 0 the update vanishes, which is
    LoRA's initialisation.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.5, printed
    p. 151.

    Examples
    --------
    >>> out = kamath_ch4_lora_forward([[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0], [0.0]], [[1.0, 0.0]], [1.0, 2.0])
    >>> out["h"]
    [2.0, 2.0]
    >>> out["r"], out["delta_h"]
    (1, [1.0, 0.0])
    """
    W0 = np.atleast_2d(np.asarray(W_0, dtype=float))
    Bm = np.atleast_2d(np.asarray(B, dtype=float))
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    if xv.ndim != 1:
        raise ValueError("x must be a vector.")
    if W0.shape[1] != xv.shape[0]:
        raise ValueError(
            f"W_0 has {W0.shape[1]} columns but x has {xv.shape[0]} "
            "entries.")
    if Am.shape[1] != xv.shape[0]:
        raise ValueError(
            f"A has {Am.shape[1]} columns but x has {xv.shape[0]} entries.")
    if Bm.shape[1] != Am.shape[0]:
        raise ValueError(
            f"B is {Bm.shape} and A is {Am.shape}; the rank dimension r "
            "must match.")
    if Bm.shape[0] != W0.shape[0]:
        raise ValueError(
            f"B produces {Bm.shape[0]} outputs but W_0 produces "
            f"{W0.shape[0]}.")
    base = W0 @ xv
    delta = Bm @ (Am @ xv)
    return RichResult(payload={
        "h": [float(v) for v in base + delta],
        "base": [float(v) for v in base],
        "delta_h": [float(v) for v in delta],
        "r": int(Bm.shape[1]),
        "delta_W_rank": int(np.linalg.matrix_rank(Bm @ Am)),
        "estimate": float((base + delta)[0]), "n": int(xv.shape[0]),
        "method": "LoRA forward pass (Kamath Eq 4.5)"})


def cheatsheet():
    return "km058: h = W_0 x + B A x, rank-r update"
