# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic INT8 quantization of a matrix product."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_dynamic_quantization"]

_METHOD = "Dynamic INT8 quantization (static weights, per-batch activations)"


def geron_dynamic_quantization(x, w):
    r"""Quantize activations at runtime, weights once, multiply in INT32.

    Per batch:

    .. math::
        s_x = \frac{\max|x|}{127},\quad x_q = \mathrm{round}(x/s_x),
        \qquad
        s_w = \frac{\max|w|}{127},\quad w_q = \mathrm{round}(w/s_w)

    accumulate :math:`x_q w_q` in INT32, then dequantize by
    :math:`s_x s_w`.

    "Dynamic" means the activation scale is measured on the batch in
    front of you rather than calibrated in advance -- which is why this
    scheme needs no calibration dataset, and why its accuracy depends on
    the batch: one outlier activation stretches ``s_x`` and coarsens
    every other value in the batch.  ``max_abs_error`` against the
    float result makes that visible.

    Accumulating in INT32 is not optional bookkeeping either: a
    thousand products of two INT8 values reach ~2e7, far past INT16.

    Parameters
    ----------
    x : array-like, shape (m, k) or (k,)
        Activations.
    w : array-like, shape (k, n)
        Weights.

    Returns
    -------
    RichResult
        Payload keys ``output`` (dequantized), ``reference`` (float32
        product), ``max_abs_error``, ``relative_error``, ``scale_x``,
        ``scale_w``, ``x_quantized``, ``w_quantized``,
        ``accumulator_max``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Appendix B, Dynamic Quantization section.

    Examples
    --------
    A value that sits exactly at the scale maximum round-trips
    perfectly:

    >>> r = geron_dynamic_quantization([[1.0]], [[1.0]])
    >>> r["output"]
    [[1.0]]
    >>> r["x_quantized"]
    [[127]]
    >>> r["max_abs_error"]
    0.0

    An outlier stretches the scale and coarsens everything else -- the
    small value now lands on a quantization grid 100 times wider:

    >>> r2 = geron_dynamic_quantization([[1.0, 100.0]], [[1.0], [0.0]])
    >>> r2["x_quantized"]
    [[1, 127]]
    >>> round(r2["output"][0][0], 6)
    0.787402
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    W = np.atleast_2d(np.asarray(w, dtype=float))
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError(f"x and w must be 2-D, got {X.shape} and {W.shape}.")
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"x has {X.shape[1]} columns but w has {W.shape[0]} rows.")
    if X.size == 0 or W.size == 0:
        raise ValueError("x and w must be non-empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(W)):
        raise ValueError("x and w must be finite.")
    mx, mw = float(np.max(np.abs(X))), float(np.max(np.abs(W)))
    if mx == 0 or mw == 0:
        raise ValueError(
            "x or w is all zeros, so its INT8 scale would be 0 and the "
            "dequantization divides by zero."
        )

    sx, sw = mx / 127.0, mw / 127.0
    Xq = np.rint(X / sx).astype(np.int32)
    Wq = np.rint(W / sw).astype(np.int32)
    acc = Xq @ Wq                                  # INT32 accumulation
    out = acc.astype(float) * sx * sw
    ref = X @ W
    err = float(np.max(np.abs(out - ref)))
    denom = float(np.max(np.abs(ref)))

    return RichResult(
        title="Dynamic INT8 quantization",
        summary_lines=[("scale_x", sx), ("scale_w", sw),
                       ("Max abs error", err)],
        payload={
            "output": out.tolist(),
            "reference": ref.tolist(),
            "max_abs_error": err,
            "relative_error": float(err / denom) if denom > 0 else 0.0,
            "scale_x": sx,
            "scale_w": sw,
            "x_quantized": Xq.tolist(),
            "w_quantized": Wq.tolist(),
            "accumulator_max": int(np.max(np.abs(acc))),
            "estimate": out.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdyq: s = max|.|/127 per batch, INT32 accumulate, dequant by s_x s_w; outliers coarsen the grid"
