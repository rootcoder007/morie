# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""QLoRA: NF4-quantised frozen base weights with full-precision LoRA
adapters."""

import numpy as np

from ._richresult import RichResult
from .kmlora import kamath_lora_weight_update
from .kmnf4 import kamath_nf4_datatype

__all__ = ["kamath_qlora_4bit", "dequantize_nf4"]


def dequantize_nf4(codes, absmax, n_bins=16):
    """W = grid[codes] * absmax, with the normalised NF4 grid from
    ``morie.fn.kmnf4``. ``absmax`` is a scalar or one value per row
    (blockwise quantisation)."""
    codes = np.atleast_2d(np.asarray(codes))
    if not np.issubdtype(codes.dtype, np.integer):
        if not np.all(codes == np.round(codes)):
            raise ValueError("NF4 codes must be integers.")
        codes = codes.astype(int)
    grid = np.asarray(kamath_nf4_datatype(n_bins)["normalized"], dtype=float)
    if np.any((codes < 0) | (codes >= grid.size)):
        raise ValueError(
            f"every code must lie in [0, {grid.size - 1}]; the NF4 grid "
            f"has {grid.size} levels.")
    s = np.asarray(absmax, dtype=float)
    if s.ndim == 0:
        scale = float(s)
        if scale <= 0:
            raise ValueError("absmax must be positive.")
        return grid[codes] * scale
    s = s.ravel()
    if s.size != codes.shape[0]:
        raise ValueError(
            f"blockwise absmax needs one value per row: {s.size} for "
            f"{codes.shape[0]} rows.")
    if np.any(s <= 0):
        raise ValueError("every absmax block scale must be positive.")
    return grid[codes] * s[:, None]


def kamath_qlora_4bit(W0_nf4, A, B, alpha, r, x, n_bins=16):
    """h = Dequant_NF4(W_0_q) x + (alpha/r) B A x.

    ``W0_nf4`` is the quantised base: a dict with ``codes`` and
    ``absmax`` (or a ``(codes, absmax)`` pair). The adapter half is
    identical to plain LoRA, so it is DELEGATED to
    ``morie.fn.kmlora``; the only thing QLoRA changes is where W_0
    comes from. The dequantisation error against a supplied
    full-precision reference is reported when one is given -- that
    error is the entire question QLoRA has to answer.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, QLoRA (Dettmers
    et al. 2023).

    Examples
    --------
    >>> W0q = {"codes": [[15, 0], [0, 15]], "absmax": 2.0}
    >>> out = kamath_qlora_4bit(W0q, [[1.0, 0.0]], [[0.0], [1.0]],
    ...                         alpha=1.0, r=1, x=[1.0, 0.0])
    >>> out["W0_dequantized"]
    [[2.0, -2.0], [-2.0, 2.0]]
    >>> out["h"]
    [2.0, -1.0]
    """
    if isinstance(W0_nf4, dict):
        if "codes" not in W0_nf4 or "absmax" not in W0_nf4:
            raise ValueError(
                "W0_nf4 must carry 'codes' and 'absmax'; a 4-bit tensor "
                "without its scale cannot be dequantised.")
        codes, absmax = W0_nf4["codes"], W0_nf4["absmax"]
    else:
        try:
            codes, absmax = W0_nf4
        except (TypeError, ValueError):
            raise ValueError(
                "W0_nf4 must be a dict with 'codes'/'absmax' or a "
                "(codes, absmax) pair.") from None
    W0 = dequantize_nf4(codes, absmax, n_bins=n_bins)
    base = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    return RichResult(payload={
        "h": base["h"], "base": base["base"], "delta": base["delta"],
        "W0_dequantized": [[float(v) for v in row] for row in W0],
        "scaling": base["scaling"], "rank": base["rank"],
        "n_trainable": base["n_trainable"],
        "n_frozen_4bit": int(np.asarray(codes).size),
        "estimate": base["estimate"], "n": base["n"],
        "method": "QLoRA: NF4 dequantised base + LoRA adapter "
                  "(delegates to kmnf4 and kmlora)"})


def cheatsheet():
    return "kmqlor: kmnf4 grid dequantises W0, kmlora adds (alpha/r)BA x"
