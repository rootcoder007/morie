# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fully convolutional network (FCN) for dense prediction."""

from . import _array_core as np

from ._richresult import RichResult
from .grcvf import geron_conv2d_forward

__all__ = ["geron_fcn"]


def geron_fcn(image, model, upsample=1, activation="relu"):
    """
    Fully convolutional network (FCN) for dense prediction.

    Formula: replace FC layers with conv layers; spatial output

    A real forward pass, not a description. ``model`` is a sequence of
    convolution kernels, each ``(kernels, bias, stride)`` or just an
    array; every layer is run through
    :func:`morie.fn.grcvf.geron_conv2d_forward` (delegated), and the
    final layer produces one channel per class *at every spatial
    position* rather than one vector for the whole image.

    That is the entire FCN idea, and the arithmetic makes it concrete: a
    dense layer over a flattened ``C x H x W`` map is exactly a
    convolution with a ``C x H x W`` kernel, so converting it costs
    nothing and lets a network trained on small crops run on any larger
    image. ``receptive_field`` and ``stride_total`` report how coarse the
    resulting prediction is; ``upsample`` then scales the class map back
    up by nearest-neighbour interpolation, which is the transposed
    convolution's cheap stand-in.

    Parameters
    ----------
    image : array-like, shape (H, W) or (C, H, W)
    model : sequence
        Layers; each is ``kernels`` (shape ``(F, C, kh, kw)`` or
        ``(F, kh, kw)``) or a tuple ``(kernels, bias, stride)``.
    upsample : int, default 1
        Nearest-neighbour upsampling factor applied to the class map.
    activation : {"relu", "identity"}, default "relu"
        Applied after every layer but the last.

    Returns
    -------
    result : RichResult
        Keys: class_map, scores, segmentation, out_shape, n_classes,
        receptive_field, stride_total, upsampled_shape, estimate, n,
        method.

    Examples
    --------
    A single 1x1 two-class layer scores every pixel independently, so the
    output is a full-resolution segmentation:

    >>> img = [[1.0, -1.0], [2.0, 0.0]]
    >>> model = [np.array([[[[1.0]]], [[[-1.0]]]])]
    >>> r = geron_fcn(img, model)
    >>> r["out_shape"]
    (2, 2, 2)
    >>> r["segmentation"]
    [[0, 1], [0, 0]]

    A 2x2 kernel shrinks the map, which is the resolution loss FCNs have
    to undo:

    >>> model2 = [np.array([[[[1.0, 1.0], [1.0, 1.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])]
    >>> r2 = geron_fcn(img, model2)
    >>> r2["out_shape"]
    (2, 1, 1)
    >>> r2["class_map"][0]
    [[2.0]]

    Upsampling restores the input resolution:

    >>> r3 = geron_fcn(img, model2, upsample=2)
    >>> r3["upsampled_shape"]
    (2, 2, 2)
    >>> r3["segmentation"]
    [[0, 0], [0, 0]]

    References
    ----------
    Géron Ch 12
    """
    X = np.asarray(image, dtype=float)
    if X.ndim == 2:
        X = X[None, :, :]
    if X.ndim != 3 or X.size == 0:
        raise ValueError(f"geron_fcn: image must be (H, W) or (C, H, W), got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_fcn: image contains non-finite values")
    if model is None or len(model) == 0:
        raise ValueError("geron_fcn: model must contain at least one convolutional layer")
    if activation not in ("relu", "identity"):
        raise ValueError(f"geron_fcn: activation must be 'relu' or 'identity', got {activation!r}")
    up = int(upsample)
    if up < 1:
        raise ValueError(f"geron_fcn: upsample must be >= 1, got {upsample!r}")

    cur = X
    rf = 1
    stride_total = 1
    shapes = []
    for li, layer in enumerate(model):
        if isinstance(layer, (tuple, list)) and len(layer) == 3:
            kernels, bias, stride = layer
        else:
            kernels, bias, stride = layer, 0.0, 1
        K = np.asarray(kernels, dtype=float)
        if K.ndim == 3:
            K = K[:, None, :, :]
        if K.ndim != 4:
            raise ValueError(f"geron_fcn: layer {li} kernels must be 3-D or 4-D, got shape {K.shape}")
        if K.shape[1] != cur.shape[0]:
            raise ValueError(
                f"geron_fcn: layer {li} expects {K.shape[1]} input channels but received {cur.shape[0]}"
            )
        b = np.atleast_1d(np.asarray(bias, dtype=float))
        if b.size == 1:
            b = np.repeat(b, K.shape[0])
        if b.size != K.shape[0]:
            raise ValueError(f"geron_fcn: layer {li} bias has {b.size} entries but there are {K.shape[0]} filters")
        st = int(stride)
        if st < 1:
            raise ValueError(f"geron_fcn: layer {li} stride must be >= 1, got {stride!r}")

        maps = []
        for f in range(K.shape[0]):
            out = geron_conv2d_forward(cur, K[f], b=float(b[f]), stride=st, padding=0)
            maps.append(np.asarray(out["Y"], dtype=float))
        cur = np.stack(maps, axis=0)
        shapes.append(tuple(int(v) for v in cur.shape))
        rf += (K.shape[2] - 1) * stride_total
        stride_total *= st
        if li < len(model) - 1 and activation == "relu":
            cur = np.maximum(cur, 0.0)

    scores = cur
    seg = scores.argmax(axis=0)
    up_map = np.repeat(np.repeat(scores, up, axis=1), up, axis=2) if up > 1 else scores
    up_seg = up_map.argmax(axis=0)

    return RichResult(
        title="Fully convolutional network",
        summary_lines=[("Output", tuple(int(v) for v in scores.shape)), ("Classes", int(scores.shape[0])), ("Stride", stride_total)],
        interpretation="A dense layer is a convolution with a full-size kernel, so an FCN runs on any input resolution.",
        payload={
            "class_map": scores.tolist(),
            "scores": scores.tolist(),
            "segmentation": up_seg.tolist(),
            "coarse_segmentation": seg.tolist(),
            "out_shape": tuple(int(v) for v in scores.shape),
            "upsampled_shape": tuple(int(v) for v in up_map.shape),
            "layer_shapes": shapes,
            "n_classes": int(scores.shape[0]),
            "receptive_field": int(rf),
            "stride_total": int(stride_total),
            "upsample": up,
            "estimate": float(scores.mean()),
            "n": int(scores[0].size),
            "method": "fully convolutional forward pass; each convolution delegated to grcvf",
        },
    )


def cheatsheet():
    return "hmfcn: Fully convolutional network (FCN) for dense prediction"
