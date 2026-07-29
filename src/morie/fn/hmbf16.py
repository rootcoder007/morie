# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Brain floating point (BF16): FP32-range with FP16-size."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bf16"]

_MANTISSA_BITS = 7
_EXPONENT_BITS = 8


def geron_bf16(x, rounding="nearest_even"):
    """
    Brain floating point (BF16): FP32-range with FP16-size.

    Formula: 1 sign, 8 exponent, 7 mantissa bits

    Quantises float32 inputs to the BF16 grid by keeping the top 16 bits
    of the IEEE-754 binary32 encoding. Default rounding is round-to-nearest
    with ties-to-even, exactly as hardware BF16 conversion does;
    ``rounding="truncate"`` drops the low bits instead.

    Parameters
    ----------
    x : array-like
        Values to quantise. Cast to float32 first, so inputs outside the
        float32 range already overflow to +/-inf before quantisation.
    rounding : {"nearest_even", "truncate"}
        Rounding mode applied to the discarded 16 low mantissa bits.

    Returns
    -------
    result : RichResult
        Keys: values, bits, abs_error, max_rel_error, estimate, n, method.

    Examples
    --------
    >>> r = geron_bf16([1.0, 1.1, -1.1])
    >>> [float(v) for v in r["values"]]
    [1.0, 1.1015625, -1.1015625]
    >>> float(geron_bf16([1.1], rounding="truncate")["values"][0])
    1.09375
    >>> r["bits"][0]
    '0011111110000000'
    >>> float(geron_bf16([0.0])["values"][0])
    0.0

    References
    ----------
    Géron Appendix B
    """
    if rounding not in ("nearest_even", "truncate"):
        raise ValueError(f"geron_bf16: rounding must be 'nearest_even' or 'truncate', got {rounding!r}")
    arr = np.atleast_1d(np.asarray(x, dtype=np.float32))
    if arr.size == 0:
        raise ValueError("geron_bf16: input is empty")

    u = arr.view(np.uint32).astype(np.uint64)
    if rounding == "nearest_even":
        lsb = (u >> np.uint64(16)) & np.uint64(1)
        u = u + np.uint64(0x7FFF) + lsb
    top = (u & np.uint64(0xFFFF0000)).astype(np.uint64)
    # A NaN input must stay NaN: rounding can carry into the exponent and
    # turn a quiet NaN payload into infinity, so restore those lanes.
    out_bits = top.astype(np.uint32)
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        out_bits[nan_mask] = arr.view(np.uint32)[nan_mask] & np.uint32(0xFFFF0000)
    quant = out_bits.view(np.float32)

    finite = np.isfinite(arr) & np.isfinite(quant)
    abs_err = np.zeros(arr.shape, dtype=float)
    abs_err[finite] = np.abs(quant[finite].astype(float) - arr[finite].astype(float))
    nz = finite & (arr != 0)
    rel = np.zeros(arr.shape, dtype=float)
    rel[nz] = abs_err[nz] / np.abs(arr[nz].astype(float))
    max_rel = float(np.max(rel)) if np.any(nz) else 0.0

    bits = [format(int(b) >> 16, "016b") for b in out_bits]

    return RichResult(
        title="BF16 quantisation",
        summary_lines=[("Values", arr.size), ("Max relative error", max_rel)],
        payload={
            "values": quant.astype(np.float32),
            "bits": bits,
            "abs_error": abs_err,
            "rel_error": rel,
            "max_rel_error": max_rel,
            "mantissa_bits": _MANTISSA_BITS,
            "exponent_bits": _EXPONENT_BITS,
            "rounding": rounding,
            "estimate": float(quant[0]),
            "n": int(arr.size),
            "method": "BF16 quantisation (1 sign / 8 exponent / 7 mantissa bits)",
        },
    )


def cheatsheet():
    return "hmbf16: Brain floating point (BF16): FP32-range with FP16-size"
