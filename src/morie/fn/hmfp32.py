# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-precision (FP32) representation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_fp32"]

_EXP_BITS = 8
_MAN_BITS = 23
_BIAS = 127


def geron_fp32(x):
    """
    Single-precision (FP32) representation.

    Formula: 1 sign, 8 exponent, 23 mantissa bits

    The bit fields are recovered from the actual IEEE-754 encoding by
    reinterpreting the float32 as a uint32 -- not by an arithmetic
    approximation -- so subnormals, infinities and NaN are classified
    correctly. The reconstructed value ``(-1)^s * 2^(e-127) * (1 + m/2^23)``
    is returned and equals the stored float32 exactly.

    ``eps`` is ``2^-23``, the gap between 1 and the next representable
    number, and ``rel_error`` is the cost of storing the input at this
    precision.

    Parameters
    ----------
    x : array-like
        Values to encode.

    Returns
    -------
    result : RichResult
        Keys: value, sign, exponent, mantissa, exponent_field,
        mantissa_field, reconstructed, rel_error, eps, kind, estimate,
        n, method.

    Examples
    --------
    1.0 is sign 0, biased exponent 127, zero mantissa:

    >>> r = geron_fp32([1.0])
    >>> r["sign"], r["exponent_field"], r["mantissa_field"]
    ([0], [127], [0])
    >>> r["exponent"]
    [0]

    0.1 is not representable, and the relative error is about 1.5e-8:

    >>> r2 = geron_fp32([0.1])
    >>> r2["rel_error"][0] > 0
    True
    >>> r2["rel_error"][0] < 1e-7
    True

    The machine epsilon and the classification of specials:

    >>> round(r["eps"], 12)
    1.19209e-07
    >>> geron_fp32([1e40, 0.0, 1e-42])["kind"]
    ['inf', 'zero', 'subnormal']

    References
    ----------
    Géron Appendix B
    """
    a = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if a.size == 0:
        raise ValueError("geron_fp32: x is empty")
    with np.errstate(over="ignore"):
        f = a.astype(np.float32)
    bits = f.view(np.uint32) if f.flags.c_contiguous else np.ascontiguousarray(f).view(np.uint32)

    sign = ((bits >> 31) & 0x1).astype(int)
    ef = ((bits >> _MAN_BITS) & 0xFF).astype(int)
    mf = (bits & ((1 << _MAN_BITS) - 1)).astype(int)

    kind = []
    recon = np.empty(a.size, dtype=np.float64)
    for i in range(a.size):
        s, e, m = int(sign[i]), int(ef[i]), int(mf[i])
        if e == 0xFF:
            kind.append("nan" if m else "inf")
            recon[i] = np.nan if m else (-np.inf if s else np.inf)
        elif e == 0:
            kind.append("zero" if m == 0 else "subnormal")
            recon[i] = (-1.0) ** s * 2.0 ** (1 - _BIAS) * (m / 2**_MAN_BITS)
        else:
            kind.append("normal")
            recon[i] = (-1.0) ** s * 2.0 ** (e - _BIAS) * (1.0 + m / 2**_MAN_BITS)

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(a != 0, np.abs(f.astype(np.float64) - a) / np.abs(np.where(a == 0, 1.0, a)), 0.0)
    rel = np.where(np.isfinite(rel), rel, np.inf)

    return RichResult(
        title="FP32 representation",
        summary_lines=[("Bits", "1 + 8 + 23"), ("eps", float(2.0**-_MAN_BITS))],
        interpretation="FP32 keeps ~7 decimal digits; the exponent range is 2^-126 to about 3.4e38.",
        payload={
            "value": f.astype(float).tolist(),
            "sign": sign.tolist(),
            "exponent": (ef - _BIAS).tolist(),
            "exponent_field": ef.tolist(),
            "mantissa": (mf / 2**_MAN_BITS).tolist(),
            "mantissa_field": mf.tolist(),
            "reconstructed": recon.tolist(),
            "rel_error": rel.tolist(),
            "eps": float(2.0**-_MAN_BITS),
            "max_normal": float(np.finfo(np.float32).max),
            "min_normal": float(np.finfo(np.float32).tiny),
            "kind": kind,
            "bits_total": 32,
            "estimate": float(np.max(rel[np.isfinite(rel)])) if np.any(np.isfinite(rel)) else float("inf"),
            "n": int(a.size),
            "method": "IEEE-754 binary32 field decomposition via a uint32 bit view",
        },
    )


def cheatsheet():
    return "hmfp32: Single-precision (FP32) representation"
