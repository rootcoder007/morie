# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BF16 representation: 1 sign + 8 exponent + 7 mantissa bits (FP32's exponent range)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bf16_range"]

_METHOD = "bfloat16 round-trip and range analysis"

# bfloat16 is fp32 with the low 16 mantissa bits dropped, so the
# normal range is identical to fp32's.
_BF16_MAX = float(np.float32(3.3895314e38))
_BF16_MIN_NORMAL = float(np.finfo(np.float32).tiny)
_BF16_EPS = 2.0**-7  # 7 explicit mantissa bits


def _to_bf16(x32):
    """Round float32 to bfloat16 precision, ties-to-even, staying in float32."""
    u = x32.view(np.uint32)
    # Round-to-nearest-even on the 16-bit boundary.
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = ((u + bias) & np.uint32(0xFFFF0000)).astype(np.uint32)
    out = rounded.view(np.float32)
    # NaN/inf pass through unchanged apart from mantissa truncation.
    return out


def geron_bf16_range(x):
    r"""Round values to bfloat16 and report what that costs.

    bfloat16 keeps all 8 exponent bits of fp32 and throws away 16 of the
    23 mantissa bits.  The dynamic range is therefore *identical* to
    fp32 -- roughly :math:`1.18\times10^{-38}` to
    :math:`3.4\times10^{38}` -- while relative precision drops to
    :math:`2^{-8} \approx 0.4\%`.  That trade is why bf16 needs no loss
    scaling in mixed-precision training, unlike fp16, whose narrow
    exponent overflows on ordinary gradients.

    Parameters
    ----------
    x : array-like
        Values to round.

    Returns
    -------
    RichResult
        Payload keys ``bf16``, ``abs_error``, ``rel_error``,
        ``max_rel_error``, ``machine_eps``, ``max_normal``,
        ``min_normal``, ``n_overflow``, ``n_underflow``, ``exact``,
        ``estimate`` (max relative error), ``n``, ``method``.

    References
    ----------
    Géron Appendix B, BF16 section.

    Examples
    --------
    Values needing no more than 7 mantissa bits survive exactly:

    >>> r = geron_bf16_range([1.0, 1.0078125, -2.0])
    >>> r["bf16"]
    [1.0, 1.0078125, -2.0]
    >>> r["exact"]
    [True, True, True]

    ``1 + 1/256`` sits exactly halfway between two bf16 values, so
    ties-to-even sends it down to 1.0:

    >>> r2 = geron_bf16_range([1.00390625])
    >>> r2["bf16"]
    [1.0]
    >>> round(r2["max_rel_error"], 8)
    0.00389105
    >>> round(r2["machine_eps"], 8)
    0.0078125
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("x is empty.")
    if np.any(np.isnan(x)):
        raise ValueError("x contains NaN; bf16 rounding of NaN is not meaningful here.")
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    b = _to_bf16(x32)

    x64 = x32.astype(np.float64)
    b64 = b.astype(np.float64)
    abs_err = np.abs(x64 - b64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(x64 != 0, abs_err / np.abs(x64), 0.0)
    rel = np.where(np.isfinite(rel), rel, 0.0)

    finite = np.isfinite(x64)
    n_over = int(np.sum(np.abs(x64[finite]) > _BF16_MAX))
    n_under = int(np.sum((np.abs(x64[finite]) > 0) & (np.abs(x64[finite]) < _BF16_MIN_NORMAL)))

    return RichResult(
        title="bfloat16 rounding",
        summary_lines=[
            ("Max relative error", float(rel.max())),
            ("bf16 machine epsilon", _BF16_EPS),
        ],
        interpretation=(
            "bf16 shares fp32's exponent range, so overflow is rare; the cost is "
            "precision, about 2-3 decimal digits."
        ),
        payload={
            "bf16": b64.tolist(),
            "abs_error": abs_err.tolist(),
            "rel_error": rel.tolist(),
            "max_rel_error": float(rel.max()),
            "machine_eps": _BF16_EPS,
            "max_normal": _BF16_MAX,
            "min_normal": _BF16_MIN_NORMAL,
            "n_overflow": n_over,
            "n_underflow": n_under,
            "exact": (abs_err == 0).tolist(),
            "estimate": float(rel.max()),
            "n": int(x.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbf16: bf16 = fp32 exponent range with 7 mantissa bits; round-trip error report"


# compact alias per ledger/NAMING.md
geronbf16range = geron_bf16_range
