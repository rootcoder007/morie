# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 4: QLoRA double quantization of the scale constants."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_double_quantization"]


def kamath_double_quantization(scales_fp32, bits=8):
    r"""Quantize the per-block FP32 scales to int8 with a shared constant.

    The first quantization leaves one FP32 scale per weight block;
    double quantization quantizes THOSE, with a single shared FP32
    constant ``c = max|scale| / (2^{bits-1} - 1)``. Returned are the
    integer codes, the shared constant, the dequantized scales and the
    exact reconstruction error -- lossy compression whose loss is
    measured, not assumed away. Memory saved is reported in bits per
    block.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Double
    Quantization; Dettmers et al. (2023).

    Examples
    --------
    >>> out = kamath_double_quantization([1.0, 0.5, 0.25])
    >>> out["scales_int8"]
    [127, 64, 32]
    >>> round(out["max_abs_error"], 6)   # 64/127 - 0.5
    0.003937
    """
    s = np.atleast_1d(np.asarray(scales_fp32, dtype=float))
    if s.size == 0:
        raise ValueError("no quantization constants were given.")
    if not np.all(np.isfinite(s)):
        raise ValueError("the scales must be finite.")
    b = int(bits)
    if b < 2 or b > 16:
        raise ValueError(f"bits must lie in [2, 16]; got {b}.")
    qmax = 2 ** (b - 1) - 1
    peak = float(np.max(np.abs(s)))
    if peak == 0:
        raise ValueError("every scale is 0, so the shared constant is "
                         "0 and the codes are undefined.")
    c = peak / qmax
    codes = np.clip(np.round(s / c), -qmax - 1, qmax).astype(int)
    deq = codes * c
    err = np.abs(deq - s)
    return RichResult(payload={
        "estimate": float(err.max()),
        "scales_int8": [int(v) for v in codes], "shared_const": c,
        "dequantized": [float(v) for v in deq],
        "max_abs_error": float(err.max()),
        "bits_saved_per_block": 32 - b, "n": int(s.size),
        "method": "double quantization of the scale constants "
                  "(Kamath Ch 4)"})


def cheatsheet():
    return "kmdbq: int8 codes for the block scales plus one shared FP32 c"
