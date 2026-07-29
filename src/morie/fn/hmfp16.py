# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FP16 half precision."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_fp16_quant"]

_MAN_BITS = 10
_BIAS = 15


def geron_fp16_quant(x):
    """
    FP16 half precision.

    Formula: float16 representation (1 sign, 5 exp, 10 mantissa)

    Values are round-tripped through the real IEEE-754 binary16 encoding
    and the fields are read back from the uint16 bit pattern, so the two
    failure modes that make FP16 training hard are visible rather than
    hidden: ``overflowed`` marks finite inputs that became ``inf`` (above
    65504) and ``underflowed`` marks non-zero inputs that flushed to zero
    or fell into the subnormal range (below 2^-14).

    That narrow exponent range -- not the 10-bit mantissa -- is why mixed
    precision keeps a FP32 master copy of the weights and scales the loss.

    Parameters
    ----------
    x : array-like
        Values to cast.

    Returns
    -------
    result : RichResult
        Keys: value, sign, exponent, mantissa_field, rel_error,
        max_rel_error, overflowed, underflowed, eps, max_normal,
        estimate, n, method.

    Examples
    --------
    1.0 is exact; the next representable number above it is 1 + 2^-10:

    >>> r = geron_fp16_quant([1.0])
    >>> r["value"], r["rel_error"]
    ([1.0], [0.0])
    >>> round(r["eps"], 12)
    0.0009765625
    >>> r["exponent"], r["mantissa_field"]
    ([0], [0])

    0.1 costs about 2.4e-4 of relative error:

    >>> round(geron_fp16_quant([0.1])["rel_error"][0], 9)
    0.000244141

    The range limits bite well before the precision does:

    >>> r2 = geron_fp16_quant([70000.0, 1e-9])
    >>> r2["overflowed"], r2["underflowed"]
    ([True, False], [False, True])
    >>> r2["max_normal"]
    65504.0

    References
    ----------
    Géron Ch 17
    """
    a = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if a.size == 0:
        raise ValueError("geron_fp16_quant: x is empty")
    with np.errstate(over="ignore"):
        h = a.astype(np.float16)
    bits = np.ascontiguousarray(h).view(np.uint16)

    sign = ((bits >> 15) & 0x1).astype(int)
    ef = ((bits >> _MAN_BITS) & 0x1F).astype(int)
    mf = (bits & ((1 << _MAN_BITS) - 1)).astype(int)

    back = h.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(a != 0, np.abs(back - a) / np.abs(np.where(a == 0, 1.0, a)), 0.0)
    over = [bool(np.isfinite(a[i]) and not np.isfinite(back[i])) for i in range(a.size)]
    under = [bool(a[i] != 0 and (back[i] == 0 or (ef[i] == 0 and mf[i] != 0))) for i in range(a.size)]
    finite_rel = rel[np.isfinite(rel)]

    return RichResult(
        title="FP16 half precision",
        summary_lines=[("Bits", "1 + 5 + 10"), ("Max normal", 65504.0), ("eps", float(2.0**-_MAN_BITS))],
        warnings=(["some values overflowed to inf in FP16"] if any(over) else [])
        + (["some values underflowed to zero or subnormal in FP16"] if any(under) else []),
        interpretation="FP16's exponent range, not its 10-bit mantissa, is what forces loss scaling.",
        payload={
            "value": [float(v) for v in back],
            "sign": sign.tolist(),
            "exponent": (ef - _BIAS).tolist(),
            "exponent_field": ef.tolist(),
            "mantissa_field": mf.tolist(),
            "rel_error": [float(v) for v in rel],
            "max_rel_error": float(finite_rel.max()) if finite_rel.size else float("inf"),
            "overflowed": over,
            "underflowed": under,
            "eps": float(2.0**-_MAN_BITS),
            "max_normal": 65504.0,
            "min_normal": float(np.finfo(np.float16).tiny),
            "bits_total": 16,
            "estimate": float(finite_rel.max()) if finite_rel.size else float("inf"),
            "n": int(a.size),
            "method": "IEEE-754 binary16 round-trip with field decomposition and range diagnostics",
        },
    )


def cheatsheet():
    return "hmfp16: FP16 half precision"
