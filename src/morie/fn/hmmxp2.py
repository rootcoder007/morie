# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixed-precision training: FP16 compute with FP32 master weights."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mixed_precision"]

_FP16_MAX = 65504.0
_FP16_MIN_NORMAL = 6.103515625e-05


def geron_mixed_precision(model, loss_scale=1024.0, grads=None):
    """
    Mixed-precision training: FP16 forward, FP32 master weights.

    Formula: weights in FP32; activations/grads in FP16; loss scaling

    Half precision halves the memory and roughly doubles the arithmetic
    throughput, and it breaks training in two specific ways this function
    measures rather than describes.

    Gradients UNDERFLOW: FP16's smallest normal is 6.1e-5, and typical
    gradients sit below it, so they flush to zero and the weight stops
    learning. Scaling the loss by a constant lifts the whole gradient
    distribution into range; the gradients are unscaled again in FP32
    before the update, so the maths is unchanged. The largest safe scale
    is 65504 / max|g|, and ``recommended_loss_scale`` is the power of two
    just under it -- powers of two because they are exact in binary and
    add no rounding of their own.

    Weights stay in FP32 for the other reason: an update far smaller than
    the weight's own FP16 spacing rounds away to nothing, so a master
    copy in full precision is what makes small steps accumulate at all.

    Parameters
    ----------
    model : array-like or mapping of arrays
        Master weights (FP32).
    loss_scale : float, default 1024.0
        Scale applied to the loss before backprop (positive).
    grads : array-like or mapping, optional
        Unscaled gradients, matching ``model``, for the range analysis.

    Returns
    -------
    result : RichResult
        Keys: fp16_weights, overflow, n_underflow, recommended_loss_scale,
        max_safe_loss_scale, memory_bytes_fp32, memory_bytes_fp16,
        estimate, n, method.

    Examples
    --------
    A gradient of 1e-8 is far below FP16's smallest normal and would
    flush to zero unscaled; at a scale of 1024 it still does.

    >>> r = geron_mixed_precision([1.0, 2.0], loss_scale=1024.0, grads=[1e-8, 1.0])
    >>> int(r["n_underflow"]), bool(r["overflow"])
    (1, False)

    The largest safe scale is 65504/1 = 65504, so the recommended power
    of two is 32768:

    >>> round(float(r["max_safe_loss_scale"]), 1), float(r["recommended_loss_scale"])
    (65504.0, 32768.0)

    A scale that pushes a gradient past 65504 overflows:

    >>> bool(geron_mixed_precision([1.0], loss_scale=65536.0, grads=[2.0])["overflow"])
    True

    Half precision halves the weight memory:

    >>> int(r["memory_bytes_fp32"]), int(r["memory_bytes_fp16"])
    (8, 4)

    References
    ----------
    Geron Ch 17
    """
    scale = float(loss_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"geron_mixed_precision: loss_scale must be positive and finite, got {loss_scale!r}")

    if hasattr(model, "items"):
        keys = list(model.keys())
        tensors = [np.atleast_1d(np.asarray(model[k], dtype=float)) for k in keys]
    else:
        keys = None
        tensors = [np.atleast_1d(np.asarray(model, dtype=float))]
    if sum(t.size for t in tensors) == 0:
        raise ValueError("geron_mixed_precision: model has no weights")
    for t in tensors:
        if not np.all(np.isfinite(t)):
            raise ValueError("geron_mixed_precision: model contains non-finite weights")

    half = [t.astype(np.float16) for t in tensors]
    w_over = any(bool(np.any(np.abs(t) > _FP16_MAX)) for t in tensors)

    n_under = 0
    overflow = False
    max_safe = float("inf")
    if grads is not None:
        if keys is None:
            gts = [np.atleast_1d(np.asarray(grads, dtype=float))]
        else:
            if not hasattr(grads, "items"):
                raise ValueError("geron_mixed_precision: grads must be a mapping when model is one")
            missing = [k for k in keys if k not in grads]
            if missing:
                raise ValueError(f"geron_mixed_precision: grads is missing {missing}")
            gts = [np.atleast_1d(np.asarray(grads[k], dtype=float)) for k in keys]
        for t, g in zip(tensors, gts):
            if g.shape != t.shape:
                raise ValueError(f"geron_mixed_precision: gradient shape {g.shape} does not match weight shape {t.shape}")
            if not np.all(np.isfinite(g)):
                raise ValueError("geron_mixed_precision: grads contain non-finite values")
        gmax = max(float(np.max(np.abs(g))) for g in gts)
        max_safe = _FP16_MAX / gmax if gmax > 0 else float("inf")
        for g in gts:
            s = np.abs(g) * scale
            n_under += int(np.sum((s > 0) & (s < _FP16_MIN_NORMAL)))
            if np.any(s > _FP16_MAX):
                overflow = True

    if np.isfinite(max_safe) and max_safe >= 1.0:
        rec = float(2.0 ** np.floor(np.log2(max_safe)))
    elif np.isfinite(max_safe):
        rec = float(max_safe)
    else:
        rec = scale

    nbytes = int(sum(t.size for t in tensors))
    fp16_out = half[0] if keys is None else {k: v for k, v in zip(keys, half)}
    return RichResult(
        title="Mixed-precision plan",
        summary_lines=[("Loss scale", scale), ("Underflowing gradients", n_under), ("Overflow", overflow)],
        warnings=(["weights exceed the FP16 range 65504 and cannot be cast without saturating"] if w_over else []),
        interpretation="Scale the loss to lift gradients above 6.1e-5; keep FP32 masters so small updates accumulate.",
        payload={
            "fp16_weights": fp16_out,
            "overflow": overflow,
            "weight_overflow": w_over,
            "n_underflow": int(n_under),
            "loss_scale": scale,
            "max_safe_loss_scale": max_safe,
            "recommended_loss_scale": rec,
            "memory_bytes_fp32": nbytes * 4,
            "memory_bytes_fp16": nbytes * 2,
            "fp16_max": _FP16_MAX,
            "fp16_min_normal": _FP16_MIN_NORMAL,
            "estimate": rec,
            "n": nbytes,
            "method": "Mixed-precision cast with loss-scaling range analysis",
        },
    )


def cheatsheet():
    return "hmmxp2: Mixed-precision training plan with loss scaling"
