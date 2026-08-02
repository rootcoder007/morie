# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static post-training quantization using calibration data."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_static_quantization_ptq"]


def geron_static_quantization_ptq(model, calibration_data, bits=8, percentile=100.0):
    """
    Static post-training quantization (PTQ) using calibration data.

    Formula: observe activation ranges on calibration set; pick scales/zeros

    Weights are quantized SYMMETRICALLY (zero maps to zero, so a pruned
    weight stays pruned and no bias creeps in), activations AFFINELY from
    the observed range, because a post-ReLU activation is one-sided and a
    symmetric grid would throw away half its levels.

    The calibration range is the whole game: one outlier activation
    stretches the scale and every ordinary value collapses onto a handful
    of levels. ``percentile`` clips the range at a quantile instead of
    the maximum, which trades a few saturated outliers for resolution
    where the mass is.

    Parameters
    ----------
    model : array-like or mapping of arrays
        Weight tensors.
    calibration_data : array-like
        Representative activations.
    bits : int, default 8
        Bit width, 2 to 16.
    percentile : float, default 100.0
        Range clipping percentile in (0, 100].

    Returns
    -------
    result : RichResult
        Keys: weight_scale, quantized_weights, dequantized_weights,
        activation_scale, zero_point, max_weight_error, compression,
        estimate, n, method.

    Examples
    --------
    Activations calibrated on [-1, 1] at 8 bits: 2/255 per level, and
    the zero of the float scale lands at level 128.

    >>> r = geron_static_quantization_ptq([-1.0, 0.5, 1.0], [-1.0, 0.0, 1.0])
    >>> round(float(r["activation_scale"]), 9), int(r["zero_point"])
    (0.007843137, 128)

    Weights use a symmetric grid of 127 steps, so 1.0 maps to 127 and
    0.5 to 64:

    >>> round(float(r["weight_scale"]), 9)
    0.007874016
    >>> [int(q) for q in r["quantized_weights"]]
    [-127, 64, 127]

    Round-trip error never exceeds half a step:

    >>> bool(r["max_weight_error"] <= r["weight_scale"] / 2 + 1e-15)
    True

    References
    ----------
    Geron Appendix B
    """
    b = int(bits)
    if not (2 <= b <= 16):
        raise ValueError(f"geron_static_quantization_ptq: bits must lie in [2, 16], got {bits!r}")
    pct = float(percentile)
    if not (0.0 < pct <= 100.0):
        raise ValueError(f"geron_static_quantization_ptq: percentile must lie in (0, 100], got {percentile!r}")

    if hasattr(model, "items"):
        keys = list(model.keys())
        tensors = [np.atleast_1d(np.asarray(model[k], dtype=float)) for k in keys]
    else:
        keys = None
        tensors = [np.atleast_1d(np.asarray(model, dtype=float))]
    if sum(t.size for t in tensors) == 0:
        raise ValueError("geron_static_quantization_ptq: model has no weights")
    for t in tensors:
        if not np.all(np.isfinite(t)):
            raise ValueError("geron_static_quantization_ptq: model contains non-finite weights")

    cal = np.atleast_1d(np.asarray(calibration_data, dtype=float)).ravel()
    if cal.size == 0:
        raise ValueError("geron_static_quantization_ptq: calibration_data is empty")
    if not np.all(np.isfinite(cal)):
        raise ValueError("geron_static_quantization_ptq: calibration_data contains non-finite values")

    qmax = 2 ** (b - 1) - 1
    lo = float(np.percentile(cal, (100.0 - pct) / 2.0)) if pct < 100.0 else float(cal.min())
    hi = float(np.percentile(cal, 100.0 - (100.0 - pct) / 2.0)) if pct < 100.0 else float(cal.max())
    if hi == lo:
        raise ValueError(
            f"geron_static_quantization_ptq: the calibration range is a single value ({lo}); no scale exists"
        )
    a_scale = (hi - lo) / (2**b - 1)
    zero_point = int(np.round(-lo / a_scale))

    qw, dqw, errs = [], [], []
    wmax = max(float(np.max(np.abs(t))) for t in tensors)
    if wmax == 0:
        raise ValueError("geron_static_quantization_ptq: every weight is zero; the scale is undefined")
    w_scale = wmax / qmax
    for t in tensors:
        q = np.clip(np.round(t / w_scale), -qmax, qmax)
        d = q * w_scale
        qw.append(q.astype(int))
        dqw.append(d)
        errs.append(float(np.max(np.abs(d - t))))

    if keys is None:
        qout, dout = qw[0], dqw[0]
    else:
        qout = {k: v for k, v in zip(keys, qw)}
        dout = {k: v for k, v in zip(keys, dqw)}

    return RichResult(
        title="Static post-training quantization",
        summary_lines=[("Bits", b), ("Weight scale", w_scale), ("Activation scale", a_scale)],
        interpretation="Calibration outliers stretch the activation scale; clip by percentile when they do.",
        payload={
            "quantized_weights": qout,
            "dequantized_weights": dout,
            "weight_scale": w_scale,
            "activation_scale": a_scale,
            "zero_point": zero_point,
            "activation_range": (lo, hi),
            "max_weight_error": max(errs),
            "compression": 32.0 / b,
            "bits": b,
            "estimate": qout,
            "n": int(sum(t.size for t in tensors)),
            "method": "PTQ: symmetric weight grid, affine activation grid from calibration",
        },
    )


def cheatsheet():
    return "hmptq: Static post-training quantization from calibration ranges"
