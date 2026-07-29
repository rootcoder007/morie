# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symmetric INT8 quantization."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_int8_quantization", "quantize_symmetric"]

_METHOD = "Symmetric INT8 quantization"


def quantize_symmetric(x, bits=8):
    """Return ``(q, s, dequantized)`` for a symmetric signed quantizer."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("x is empty; there is nothing to quantize.")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains non-finite values.")
    bits = int(bits)
    if bits < 2 or bits > 32:
        raise ValueError(f"bits must lie in [2, 32], got {bits}.")
    qmax = 2 ** (bits - 1) - 1
    amax = float(np.max(np.abs(a)))
    if amax == 0:
        raise ValueError("x is all zeros; the scale max|x|/qmax would be zero.")
    s = amax / qmax
    q = np.clip(np.round(a / s), -qmax, qmax)
    return q, s, q * s


def geron_int8_quantization(x, bits=8):
    r"""Map a float tensor onto the signed 8-bit grid.

    .. math::
        s = \frac{\max|x|}{127}, \qquad q = \mathrm{round}(x / s)

    Symmetric, so zero maps to zero exactly -- which matters because
    padding and ReLU outputs are full of zeros, and an asymmetric scheme
    that puts zero between grid points injects error into every one of
    them.  The range is ``[-127, 127]``, not ``[-128, 127]``: giving up
    one code keeps the grid symmetric around zero.  Per-tensor scaling
    like this is cheap and is exactly why a single outlier weight ruins
    the resolution of everything else -- the reported SNR shows it.

    Parameters
    ----------
    x : array-like
    bits : int, optional
        Bit width, default 8.

    Returns
    -------
    RichResult
        Payload keys ``q`` (integers), ``scale``, ``dequantized``,
        ``max_abs_error``, ``snr_db``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 17, Quantization / INT8 section.

    Examples
    --------
    ``max|x| = 1.0`` gives ``s = 1/127``, so 0.5 lands on code 64
    (``0.5 * 127 = 63.5``, rounded half to even by numpy):

    >>> r = geron_int8_quantization([1.0, 0.5, -1.0, 0.0])
    >>> r["q"]
    [127.0, 64.0, -127.0, 0.0]
    >>> round(r["scale"], 8)
    0.00787402

    Zero survives exactly, and the error never exceeds half a step:

    >>> r["dequantized"][3]
    0.0
    >>> r["max_abs_error"] <= r["scale"] / 2 + 1e-12
    True
    """
    q, s, deq = quantize_symmetric(x, bits)
    a = np.asarray(x, dtype=float)
    err = deq - a
    signal = float(np.sum(a**2))
    noise = float(np.sum(err**2))
    snr = float("inf") if noise == 0 else 10.0 * np.log10(signal / noise)

    return RichResult(
        title="INT8 quantization",
        summary_lines=[("Scale", s), ("SNR (dB)", snr)],
        payload={
            "q": q.tolist(),
            "scale": float(s),
            "dequantized": deq.tolist(),
            "max_abs_error": float(np.max(np.abs(err))),
            "snr_db": snr,
            "bits": int(bits),
            "estimate": q.tolist(),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grq8: s = max|x|/127, q = round(x/s) clipped to +-127; zero maps to zero exactly"
