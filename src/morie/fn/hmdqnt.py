# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic quantization: quantize weights statically, activations at runtime."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_dynamic_quantization"]

_DTYPES = {"int8": (-128, 127, True), "uint8": (0, 255, False), "int16": (-32768, 32767, True)}


def geron_dynamic_quantization(model, dtype="int8", activations=None):
    """
    Dynamic quantization: quantize weights statically, activations at runtime.

    Formula: weights: per-tensor INT8; activations: per-batch scale

    The two halves really are different, and both are computed here.
    Weights get one affine map per tensor, fixed ahead of time:
    ``q = round(w/s) + z`` with ``s = (max-min)/(qmax-qmin)``. Activations
    have no fixed range, so a scale is derived per batch at call time --
    that is the "dynamic" part, and the reason no calibration set is
    needed.

    Symmetric INT8 pins the zero point at 0 so that a zero weight stays
    exactly zero after the round trip; asymmetric UINT8 does not, which is
    reported as ``zero_point``.

    ``model`` is a mapping ``name -> weight array`` (or a single array).

    Parameters
    ----------
    model : mapping or array-like
        Weight tensors.
    dtype : {"int8", "uint8", "int16"}, default "int8"
    activations : array-like, optional
        A batch of activations to quantize dynamically.

    Returns
    -------
    result : RichResult
        Keys: quantized, scales, zero_points, dequantized, max_abs_error,
        compression, activation, estimate, n, method.

    Examples
    --------
    A tensor spanning -1 to 1 in INT8 gets a scale of 1/127 and keeps its
    extremes exactly:

    >>> r = geron_dynamic_quantization({"W": [-1.0, 0.0, 1.0]})
    >>> round(r["scales"]["W"], 12)
    0.007874015748
    >>> r["quantized"]["W"]
    [-127, 0, 127]
    >>> [round(v, 12) for v in r["dequantized"]["W"]]
    [-1.0, 0.0, 1.0]
    >>> r["compression"]
    4.0

    Zero survives the round trip exactly under symmetric quantization:

    >>> r2 = geron_dynamic_quantization({"W": [-0.3, 0.0, 0.9]})
    >>> r2["dequantized"]["W"][1]
    0.0
    >>> r2["zero_points"]["W"]
    0

    The activation scale is derived from the batch that is passed in:

    >>> r3 = geron_dynamic_quantization({"W": [1.0]}, activations=[0.0, 2.0])
    >>> round(r3["activation"]["scale"], 12)
    0.015748031496

    References
    ----------
    Géron Appendix B
    """
    if dtype not in _DTYPES:
        raise ValueError(f"geron_dynamic_quantization: dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    qmin, qmax, symmetric = _DTYPES[dtype]
    bits = 8 if dtype in ("int8", "uint8") else 16

    if isinstance(model, dict):
        tensors = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in model.items()}
    else:
        tensors = {"weight": np.atleast_1d(np.asarray(model, dtype=float))}
    if not tensors:
        raise ValueError("geron_dynamic_quantization: model contains no weight tensors")

    q, scales, zps, deq, errs = {}, {}, {}, {}, {}
    for name, W in tensors.items():
        if W.size == 0:
            raise ValueError(f"geron_dynamic_quantization: tensor {name!r} is empty")
        if not np.all(np.isfinite(W)):
            raise ValueError(f"geron_dynamic_quantization: tensor {name!r} contains non-finite values")
        if symmetric:
            amax = float(np.max(np.abs(W)))
            if amax == 0:
                raise ValueError(f"geron_dynamic_quantization: tensor {name!r} is all zeros; the scale is undefined")
            s = amax / qmax
            z = 0
        else:
            lo, hi = float(W.min()), float(W.max())
            if hi == lo:
                raise ValueError(f"geron_dynamic_quantization: tensor {name!r} is constant; the scale is undefined")
            s = (hi - lo) / (qmax - qmin)
            z = int(round(qmin - lo / s))
        qi = np.clip(np.round(W / s).astype(int) + z, qmin, qmax)
        back = (qi - z) * s
        q[name] = qi.tolist()
        scales[name] = float(s)
        zps[name] = int(z)
        deq[name] = back.tolist()
        errs[name] = float(np.max(np.abs(back - W)))

    act = None
    if activations is not None:
        A = np.atleast_1d(np.asarray(activations, dtype=float))
        if A.size == 0:
            raise ValueError("geron_dynamic_quantization: activations is empty")
        if not np.all(np.isfinite(A)):
            raise ValueError("geron_dynamic_quantization: activations contains non-finite values")
        amax = float(np.max(np.abs(A)))
        if amax == 0:
            raise ValueError("geron_dynamic_quantization: activations are all zero; the runtime scale is undefined")
        sa = amax / qmax
        qa = np.clip(np.round(A / sa).astype(int), qmin, qmax)
        act = {
            "scale": float(sa),
            "quantized": qa.tolist(),
            "dequantized": (qa * sa).tolist(),
            "max_abs_error": float(np.max(np.abs(qa * sa - A))),
            "note": "scale computed from this batch at run time, not calibrated in advance",
        }

    n_params = int(sum(v.size for v in tensors.values()))

    return RichResult(
        title="Dynamic quantization",
        summary_lines=[("dtype", dtype), ("Tensors", len(tensors)), ("Compression", float(32 / bits))],
        interpretation="Weights are mapped once; activations get a fresh scale per batch, so no calibration set is required.",
        payload={
            "quantized": q,
            "scales": scales,
            "zero_points": zps,
            "dequantized": deq,
            "max_abs_error": errs,
            "compression": float(32 / bits),
            "bits": bits,
            "dtype": dtype,
            "symmetric": symmetric,
            "activation": act,
            "estimate": float(max(errs.values())),
            "n": n_params,
            "method": "per-tensor affine weight quantization with per-batch dynamic activation scaling",
        },
    )


def cheatsheet():
    return "hmdqnt: Dynamic quantization: quantize weights statically, activations at runtime"
