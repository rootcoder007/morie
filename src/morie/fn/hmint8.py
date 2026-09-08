# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""INT8 quantization: post-training 8-bit weight+activation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_int8_quant"]

_METHOD = "Post-training integer quantization"


def geron_int8_quant(x, n_bits=8, symmetric=True):
    """
    INT8 quantization: post-training 8-bit weight+activation.

    Formula: q = round((x - z) / s); x = q*s + z

    Affine quantization onto ``2^n_bits`` integer levels.  Two schemes:

    *symmetric* -- range ``[-max|x|, max|x|]`` mapped to
    ``[-(2^(b-1) - 1), 2^(b-1) - 1]`` with zero-point 0.  Zero is
    represented exactly, which matters because padding, ReLU output and
    masked positions are all exactly zero and a scheme that rounds them
    to 0.4 of a step leaks error everywhere.

    *asymmetric* -- range ``[min, max]`` mapped to ``[0, 2^b - 1]`` with
    an integer zero-point.  It uses the whole range, so it is the better
    choice for a one-sided distribution like post-ReLU activations,
    at the cost of an extra offset per tensor.

    The dequantized tensor and the realised error are returned; the
    theoretical bound is ``s/2`` per element and the measured maximum
    should never exceed it.

    Parameters
    ----------
    x : array-like
        Tensor to quantize.
    n_bits : int
        Bit width, ``2 <= n_bits <= 16``.
    symmetric : bool
        Symmetric or asymmetric scheme.

    Returns
    -------
    result : RichResult
        Keys: q, dequantized, scale, zero_point, max_error, rel_error,
        compression, estimate, n, method.

    Examples
    --------
    Symmetric 8-bit over ``[-1, 1]``: the scale is ``1/127`` and zero
    maps to zero exactly.

    >>> r = geron_int8_quant([-1.0, 0.0, 0.5, 1.0])
    >>> round(float(r["scale"]), 12) == round(1 / 127, 12)
    True
    >>> [int(v) for v in r["q"]]
    [-127, 0, 64, 127]
    >>> float(r["zero_point"])
    0.0

    Round-trip error never exceeds half a step:

    >>> bool(r["max_error"] <= r["scale"] / 2 + 1e-15)
    True

    Asymmetric over a one-sided range uses all 256 levels:

    >>> a = geron_int8_quant([0.0, 1.0, 2.0], symmetric=False)
    >>> int(a["q"].min()), int(a["q"].max())
    (0, 255)
    >>> [round(float(v), 6) for v in a["dequantized"]]
    [0.0, 1.003922, 2.0]

    Two-bit quantization of ``[-1, 1]`` leaves only the levels -1, 0, 1,
    so 0.5 sits exactly on a tie and numpy's round-half-to-even sends it
    to 0 -- a reminder that at low bit widths half the dynamic range can
    round away:

    >>> t = geron_int8_quant([-1.0, 0.0, 0.5, 1.0], n_bits=2)
    >>> [int(v) for v in t["q"]]
    [-1, 0, 0, 1]

    A constant tensor has no range to quantize:

    >>> geron_int8_quant([3.0, 3.0], symmetric=False)
    Traceback (most recent call last):
        ...
    ValueError: geron_int8_quant: the tensor is constant (all values 3.0), so the quantization scale would be zero

    References
    ----------
    Géron Ch 17
    """
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 0:
        raise ValueError("geron_int8_quant: x is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_int8_quant: x contains non-finite values")
    b = int(n_bits)
    if not (2 <= b <= 16):
        raise ValueError(f"geron_int8_quant: n_bits must lie in 2..16, got {n_bits!r}")

    lo, hi = float(a.min()), float(a.max())
    if symmetric:
        amax = float(np.max(np.abs(a)))
        if amax == 0:
            raise ValueError(
                "geron_int8_quant: the tensor is all zeros, so the quantization scale would be zero"
            )
        qmax = 2 ** (b - 1) - 1
        scale = amax / qmax
        zero = 0.0
        q = np.clip(np.round(a / scale), -qmax, qmax).astype(np.int64)
        deq = q * scale
    else:
        if hi == lo:
            raise ValueError(
                f"geron_int8_quant: the tensor is constant (all values {lo}), so the quantization scale would be zero"
            )
        qmax = 2**b - 1
        scale = (hi - lo) / qmax
        zero = lo
        q = np.clip(np.round((a - zero) / scale), 0, qmax).astype(np.int64)
        deq = q * scale + zero

    err = np.abs(deq - a)
    denom = np.maximum(np.abs(a), np.finfo(float).tiny)
    rel = float(np.max(err / denom)) if np.any(a != 0) else 0.0
    compression = 64.0 / b

    return RichResult(
        title=f"{b}-bit quantization",
        summary_lines=[
            ("Scheme", "symmetric" if symmetric else "asymmetric"),
            ("Scale", float(scale)),
            ("Zero point", float(zero)),
            ("Max absolute error", float(np.max(err))),
            ("Compression vs float64", compression),
        ],
        interpretation=(
            "Symmetric represents zero exactly, which matters for padding and post-ReLU tensors; "
            "asymmetric uses the whole range, which matters for one-sided distributions."
        ),
        payload={
            "q": q,
            "dequantized": deq,
            "scale": float(scale),
            "zero_point": float(zero),
            "max_error": float(np.max(err)),
            "rel_error": rel,
            "compression": compression,
            "n_levels": int(2**b),
            "estimate": float(np.max(err)),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmint8: affine integer quantization q = round((x - z)/s), symmetric or asymmetric, with round-trip error"


# compact alias per ledger/NAMING.md
geronint8quant = geron_int8_quant
